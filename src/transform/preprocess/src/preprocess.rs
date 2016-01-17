
use std::error;
use std::fmt;
use slp_shared::*;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessError {
    UnknownCommand,
    InvalidInclude,
    InvalidDefine,
    MacroAlreadyDefined(String),
    MacroRequiresArguments,
    MacroArgumentsNeverEnd,
    MacroExpectsDifferentNumberOfArguments,
    FailedToFindFile,
    InvalidIf(String),
    FailedToParseIfCondition(String),
    InvalidIfndef(String),
    InvalidElse,
    InvalidEndIf,
    ConditionChainNotFinished,
    ElseNotMatched,
    EndIfNotMatched,
}

impl error::Error for PreprocessError {
    fn description(&self) -> &str {
        match *self {
            PreprocessError::UnknownCommand => "unknown preprocessor command",
            PreprocessError::InvalidInclude => "invalid #include command",
            PreprocessError::InvalidDefine => "invalid #define command",
            PreprocessError::MacroAlreadyDefined(_) => "macro already defined",
            PreprocessError::MacroRequiresArguments => "macro function requires arguments",
            PreprocessError::MacroArgumentsNeverEnd => "expected end of macro arguments",
            PreprocessError::MacroExpectsDifferentNumberOfArguments => {
                "macro requires different number of arguments"
            }
            PreprocessError::FailedToFindFile => "could not find file",
            PreprocessError::InvalidIf(_) => "invalid #if",
            PreprocessError::FailedToParseIfCondition(_) => "#if condition parser failed",
            PreprocessError::InvalidIfndef(_) => "invalid #ifndef",
            PreprocessError::InvalidElse => "invalid #else",
            PreprocessError::InvalidEndIf => "invalid #endif",
            PreprocessError::ConditionChainNotFinished => "not enough #endif's encountered",
            PreprocessError::ElseNotMatched => "encountered #else but with no matching #if",
            PreprocessError::EndIfNotMatched => "encountered #endif but with no matching #if",
        }
    }
}

impl fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", error::Error::description(self))
    }
}

pub struct PreprocessedText {
    code: Vec<u8>,
    debug_locations: LineMap,
}

impl PreprocessedText {
    fn from_intermediate_text(text: IntermediateText) -> PreprocessedText {
        PreprocessedText {
            code: text.buffer.into_bytes(),
            debug_locations: text.debug_locations,
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.code
    }
    pub fn get_file_location(&self, stream_location: &StreamLocation) -> Result<FileLocation, ()> {
        self.debug_locations.get_file_location(stream_location)
    }
}

struct IntermediateText {
    buffer: String,
    debug_locations: LineMap,
}

impl IntermediateText {
    fn new() -> IntermediateText {
        IntermediateText {
            buffer: String::new(),
            debug_locations: LineMap { lines: vec![] },
        }
    }
    fn push_str(&mut self, segment: &str, segment_location: FileLocation) {
        let parts = segment.split('\n');
        let last = parts.clone().count() - 1;
        for (index, part) in parts.enumerate() {
            let location = FileLocation(segment_location.0.clone(),
                                        Line((segment_location.1).0 + index as u64),
                                        segment_location.2.clone());
            let stream_location_in_buffer = StreamLocation(self.buffer.len() as u64);
            self.buffer.push_str(part);
            if index != last {
                self.buffer.push('\n');
            }
            self.debug_locations.lines.push((stream_location_in_buffer, location));
        }
    }
}

struct LineMap {
    lines: Vec<(StreamLocation, FileLocation)>,
}

impl LineMap {
    fn get_file_location(&self, stream_location: &StreamLocation) -> Result<FileLocation, ()> {
        let mut lower = 0;
        let mut upper = self.lines.len();
        while lower < upper - 1 {
            let next_index = (lower + upper) / 2;
            assert!(next_index > lower);
            assert!(next_index <= upper);

            let &(ref line_stream, _) = &self.lines[next_index];
            let matches = line_stream.0 <= stream_location.0;

            if matches {
                lower = next_index;
            } else {
                upper = next_index;
            }
        }
        let last_line = if lower == self.lines.len() {
            None
        } else {
            Some(lower)
        };
        match last_line {
            Some(index) => {
                let (ref line_stream, ref line_file) = self.lines[index];
                let FileLocation(base_file, base_line, base_column) = line_file.clone();
                let column = Column(base_column.0 + (stream_location.0 - line_stream.0));
                Ok(FileLocation(base_file, base_line, column))
            }
            None => Err(()),
        }
    }
}

fn is_identifier_char(c: char) -> bool {
    (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c == '_')
}

#[derive(PartialEq, Debug, Clone)]
struct MacroArg(u64);

#[derive(PartialEq, Debug, Clone)]
enum MacroSegment {
    Text(String),
    Arg(MacroArg),
}

impl MacroSegment {
    fn split(self, arg: &str, index: u64, segments: &mut Vec<MacroSegment>) {
        match self {
            MacroSegment::Text(text) => {
                match find_macro(&text, arg) {
                    Some(sz) => {
                        let before = &text[..sz];
                        let after_offset = sz + arg.len();
                        let after = &text[after_offset..];
                        assert_eq!(before.to_string() + arg + after, text);
                        if before.len() > 0 {
                            segments.push(MacroSegment::Text(before.to_string()));
                        }
                        segments.push(MacroSegment::Arg(MacroArg(index)));
                        if after.len() > 0 {
                            MacroSegment::Text(after.to_string()).split(arg, index, segments);
                        }
                        return;
                    }
                    None => {}
                }
                segments.push(MacroSegment::Text(text))
            }
            MacroSegment::Arg(arg) => segments.push(MacroSegment::Arg(arg)),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
struct Macro(String, u64, Vec<MacroSegment>, FileLocation);

impl Macro {
    fn from_definition(head: &str,
                       body: &str,
                       location: FileLocation)
                       -> Result<Macro, PreprocessError> {
        Ok(match head.find('(') {
            Some(sz) => {
                let name = &head[..sz];
                let mut arg_names = vec![];
                let mut remaining = &head[(sz + 1)..];
                loop {
                    let (sz, last) = match remaining.find(',') {
                        Some(sz) => (sz, false),
                        None => {
                            match remaining.find(")") {
                                Some(sz) => (sz, true),
                                None => return Err(PreprocessError::InvalidDefine),
                            }
                        }
                    };
                    let arg_name = &remaining[..sz];
                    let arg_name = arg_name.trim();
                    remaining = remaining[(sz + 1)..].trim_left();
                    for c in arg_name.chars() {
                        if !is_identifier_char(c) {
                            return Err(PreprocessError::InvalidDefine);
                        }
                    }
                    arg_names.push(arg_name);
                    if last {
                        if remaining.len() > 0 {
                            return Err(PreprocessError::InvalidDefine);
                        }
                        break;
                    }
                }
                let mut last_segments = vec![MacroSegment::Text(body.to_string())];
                for (index, arg_name) in arg_names.iter().enumerate() {
                    let mut next_segments = vec![];
                    for segment in last_segments {
                        segment.split(arg_name, index as u64, &mut next_segments);
                    }
                    last_segments = next_segments;
                }
                Macro(name.to_string(),
                      arg_names.len() as u64,
                      last_segments,
                      location)
            }
            None => {
                Macro(head.to_string(),
                      0,
                      vec![MacroSegment::Text(body.to_string())],
                      location)
            }
        })
    }
}

#[derive(PartialEq, Debug, Clone)]
enum SubstitutedSegment {
    Text(String, StreamLocation),
    Replaced(String, FileLocation),
}

fn find_macro(text: &str, name: &str) -> Option<usize> {
    let mut subtext = text;
    loop {
        let sz = match subtext.find(name) {
            Some(sz) => sz,
            None => return None,
        };
        let before = &subtext[..sz];
        let after_offset = sz + name.len();
        let after = &subtext[after_offset..];

        let not_separated_before = match before.chars().last() {
            Some(c) => is_identifier_char(c),
            None => false,
        };
        let not_separated_after = match after.chars().next() {
            Some(c) => is_identifier_char(c),
            None => false,
        };

        if !not_separated_before && !not_separated_after {
            let final_sz = sz + text.len() - subtext.len();
            return Some(final_sz);
        }

        let mut subtext_chars = subtext[sz..].chars();
        subtext = match subtext_chars.next() {
            Some(_) => subtext_chars.as_str(),
            None => return None,
        };
    }
}

impl SubstitutedSegment {
    fn apply(self,
             macro_def: &Macro,
             macro_defs: &[Macro],
             output: &mut Vec<SubstitutedSegment>)
             -> Result<(), PreprocessError> {
        match self {
            SubstitutedSegment::Text(text, location) => {
                match find_macro(&text, &macro_def.0) {
                    Some(sz) => {
                        let before = &text[..sz];
                        let after_offset = sz + macro_def.0.len();
                        let mut remaining = &text[after_offset..];

                        // Read macro arguments
                        let args = if macro_def.1 > 0 {
                            // Consume the starting bracket
                            let sz = match remaining.find('(') {
                                Some(sz) => {
                                    let gap = remaining[..sz].trim();
                                    if gap.len() > 0 {
                                        return Err(PreprocessError::MacroRequiresArguments);
                                    }
                                    sz
                                }
                                None => return Err(PreprocessError::MacroRequiresArguments),
                            };
                            remaining = &remaining[(sz + 1)..];

                            // Consume all the arguments
                            let mut args = vec![];
                            loop {
                                let (sz, last) = match (remaining.find(','), remaining.find(")")) {
                                    (Some(szn), Some(szl)) if szn < szl => (szn, false),
                                    (_, Some(szl)) => (szl, true),
                                    (Some(szn), None) => (szn, false),
                                    (None, None) => {
                                        return Err(PreprocessError::MacroArgumentsNeverEnd)
                                    }
                                };
                                let arg = remaining[..sz].trim();
                                args.push(arg);
                                remaining = &remaining[(sz + 1)..];
                                if last {
                                    break;
                                }
                            }
                            args
                        } else {
                            vec![]
                        };
                        let after = remaining;

                        if args.len() as u64 != macro_def.1 {
                            return Err(PreprocessError::MacroExpectsDifferentNumberOfArguments);
                        }

                        // Substitute macros inside macro arguments
                        let args = try!(args.into_iter().fold(Ok(vec![]), |vec, arg| {
                            let mut vec = try!(vec);
                            let raw_text = SubstitutedText::new(arg, StreamLocation(0));
                            let subbed_text = try!(raw_text.apply_all(macro_defs));
                            let final_text = subbed_text.resolve();
                            vec.push(final_text);
                            Ok(vec)
                        }));

                        let after_location = StreamLocation(location.0 +
                                                            (text.len() - after.len()) as u64);
                        if before.len() > 0 {
                            output.push(SubstitutedSegment::Text(before.to_string(), location));
                        }
                        let mut replaced_text = String::new();
                        for macro_segment in &macro_def.2 {
                            match *macro_segment {
                                MacroSegment::Text(ref text) => replaced_text.push_str(text),
                                MacroSegment::Arg(MacroArg(ref index)) => {
                                    replaced_text.push_str(&args[*index as usize])
                                }
                            }
                        }
                        if replaced_text.len() > 0 {
                            output.push(SubstitutedSegment::Replaced(replaced_text,
                                                                     macro_def.3.clone()));
                        }
                        if after.len() > 0 {
                            try!(SubstitutedSegment::Text(after.to_string(), after_location)
                                     .apply(macro_def, macro_defs, output));
                        }
                        return Ok(());
                    }
                    None => {}
                }
                output.push(SubstitutedSegment::Text(text, location))
            }
            SubstitutedSegment::Replaced(text, location) => {
                output.push(SubstitutedSegment::Replaced(text, location))
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct SubstitutedText(Vec<SubstitutedSegment>);

impl SubstitutedText {
    fn new(text: &str, location: StreamLocation) -> SubstitutedText {
        SubstitutedText(vec![SubstitutedSegment::Text(text.to_string(), location)])
    }

    fn apply_all(self, macro_defs: &[Macro]) -> Result<SubstitutedText, PreprocessError> {
        let length = self.0.len();
        let segments_iter = self.0.into_iter();
        let vec = segments_iter.fold(Ok(Vec::with_capacity(length)), |vec_res, segment| {
            let mut vec = try!(vec_res);
            let mut last_segments = vec![segment];
            for macro_def in macro_defs {
                let mut next_segments = Vec::with_capacity(last_segments.len());
                for substituted_segment in last_segments {
                    try!(substituted_segment.apply(macro_def, macro_defs, &mut next_segments));
                }
                last_segments = next_segments;
            }
            vec.append(&mut last_segments);
            Ok(vec)
        });
        Ok(SubstitutedText(try!(vec)))
    }

    fn store(self, intermediate_text: &mut IntermediateText, line_map: &LineMap) {
        for substituted_segment in self.0 {
            match substituted_segment {
                SubstitutedSegment::Text(text, location) => {
                    let mut remaining = &text[..];
                    let mut loc = location.0;
                    loop {
                        let (sz, last) = match remaining.find('\n') {
                            Some(sz) => (sz + 1, false),
                            None => (remaining.len(), true),
                        };
                        let before = &remaining[..sz];
                        intermediate_text.push_str(before, match line_map.get_file_location(&StreamLocation(loc)) {
                            Ok(loc) => loc,
                            Err(()) => panic!("bad file location"),
                        });
                        remaining = &remaining[sz..];
                        loc = loc + sz as u64;
                        if last {
                            break;
                        }
                    }
                }
                SubstitutedSegment::Replaced(text, location) => {
                    intermediate_text.push_str(&text, location)
                }
            }
        }
    }

    fn resolve(self) -> String {
        let mut output = String::new();
        for substituted_segment in self.0 {
            match substituted_segment {
                SubstitutedSegment::Text(text, _) |
                SubstitutedSegment::Replaced(text, _) => output.push_str(&text),
            }
        }
        output
    }
}

#[test]
fn macro_from_definition() {
    assert_eq!(Macro::from_definition("B", "0", FileLocation::none()).unwrap(),
               Macro("B".to_string(),
                     0,
                     vec![MacroSegment::Text("0".to_string())],
                     FileLocation::none()));
    assert_eq!(Macro::from_definition("B(x)", "x", FileLocation::none()).unwrap(),
               Macro("B".to_string(),
                     1,
                     vec![MacroSegment::Arg(MacroArg(0))],
                     FileLocation::none()));
    assert_eq!(Macro::from_definition("B(x,y)", "x", FileLocation::none()).unwrap(),
               Macro("B".to_string(),
                     2,
                     vec![MacroSegment::Arg(MacroArg(0))],
                     FileLocation::none()));
    assert_eq!(Macro::from_definition("B(x,y)", "y", FileLocation::none()).unwrap(),
               Macro("B".to_string(),
                     2,
                     vec![MacroSegment::Arg(MacroArg(1))],
                     FileLocation::none()));
    assert_eq!(Macro::from_definition("B(x,xy)", "(x || xy)", FileLocation::none()).unwrap(),
               Macro("B".to_string(),
                     2,
                     vec![
        MacroSegment::Text("(".to_string()),
        MacroSegment::Arg(MacroArg(0)),
        MacroSegment::Text(" || ".to_string()),
        MacroSegment::Arg(MacroArg(1)),
        MacroSegment::Text(")".to_string()),
    ],
                     FileLocation::none()));
}

#[test]
fn macro_resolve() {

    fn run(input: &str, macros: &[Macro], expected_output: &str) {
        let text = SubstitutedText::new(input, StreamLocation(0));
        let resolved_text = text.apply_all(&macros).unwrap().resolve();
        assert_eq!(resolved_text, expected_output);
    }

    run("(A || B) && BC",
        &[Macro::from_definition("B", "0", FileLocation::none()).unwrap(),
          Macro::from_definition("BC", "1", FileLocation::none()).unwrap()],
        "(A || 0) && 1");

    run("(A || B(0, 1)) && BC",
        &[Macro::from_definition("B(x, y)", "(x && y)", FileLocation::none()).unwrap(),
          Macro::from_definition("BC", "1", FileLocation::none()).unwrap()],
        "(A || (0 && 1)) && 1");
}

/// Stores the active #if blocks
struct ConditionChain(Vec<bool>);

impl ConditionChain {
    fn new() -> ConditionChain {
        ConditionChain(vec![])
    }

    fn push(&mut self, gate: bool) {
        self.0.push(gate);
    }

    fn switch(&mut self) -> Result<(), PreprocessError> {
        match self.0.pop() {
            Some(val) => {
                self.0.push(!val);
                Ok(())
            }
            None => Err(PreprocessError::ElseNotMatched),
        }
    }

    fn pop(&mut self) -> Result<(), PreprocessError> {
        match self.0.pop() {
            Some(_) => Ok(()),
            None => Err(PreprocessError::EndIfNotMatched),
        }
    }

    fn is_active(&self) -> bool {
        self.0.iter().fold(true, |acc, gate| acc && *gate)
    }
}

fn build_file_linemap(file_contents: &str, file_name: File) -> LineMap {
    let mut line_map = LineMap { lines: vec![] };
    let file_length = file_contents.len() as u64;
    let mut stream = file_contents;
    let mut current_line = 1;
    loop {
        let (sz, final_segment) = match stream.find("\n") {
            Some(sz) => (sz + 1, false),
            None => (stream.len(), true),
        };
        let length_left = stream.len() as u64;
        line_map.lines.push((StreamLocation(file_length - length_left),
                             FileLocation(file_name.clone(), Line(current_line), Column(1))));
        current_line = current_line + 1;
        stream = &stream[sz..];
        if final_segment {
            break;
        }
    }
    line_map
}

// Function to find end of definition past escaped endlines
fn find_macro_end(mut remaining: &str) -> usize {
    let initial_length = remaining.len();
    loop {
        match remaining.find('\n') {
            Some(sz) => {
                let before_c0 = &remaining[..sz];
                let before_c1 = if sz > 0 {
                    &remaining[..(sz - 1)]
                } else {
                    ""
                };
                remaining = &remaining[(sz + 1)..];
                match (before_c0.chars().last(), before_c1.chars().last()) {
                    (Some(x), _) if x == '\\' => {}
                    (Some(x), Some(y)) if x == '\r' && y == '\\' => {}
                    _ => break,
                }
            }
            None => {
                return remaining.len();
            }
        }
    }
    initial_length - remaining.len() - 1
}

fn get_normal_end(remaining: &str) -> &str {
    let len = match remaining.find('\n') {
        Some(sz) => sz + 1,
        None => remaining.len(),
    };
    &remaining[len..]
}

fn get_macro_end(remaining: &str) -> &str {
    let len = find_macro_end(remaining) + 1;
    &remaining[len..]
}

fn preprocess_command<'a>(buffer: &mut IntermediateText,
                          include_handler: &mut IncludeHandler,
                          command: &'a str,
                          location: FileLocation,
                          macros: &mut Vec<Macro>,
                          condition_chain: &mut ConditionChain)
                          -> Result<&'a str, PreprocessError> {
    let skip = !condition_chain.is_active();
    if command.starts_with("include") {
        if skip {
            return Ok(get_normal_end(command));
        }
        let next = &command[7..];
        match next.chars().next() {
            Some(' ') | Some('\t') | Some('"') | Some('<') => {
                let args = next.trim_left();
                let end = match args.chars().next() {
                    Some('"') => '"',
                    Some('<') => '>',
                    _ => return Err(PreprocessError::InvalidInclude),
                };
                let args = &args[1..];
                match args.find(end) {
                    Some(sz) => {
                        let file_name = &args[..sz];
                        if file_name.contains('\n') {
                            return Err(PreprocessError::InvalidInclude);
                        }

                        // Include the file
                        match include_handler.load(file_name) {
                            Ok(file) => {
                                try!(preprocess_file(buffer,
                                                     include_handler,
                                                     &file,
                                                     macros,
                                                     condition_chain));



                                let next = &args[(sz + 1)..];
                                let end = match next.find('\n') {
                                    Some(sz) => {
                                        // Push a new line so the last line of the include file is on a
                                        // separate line to the first line after the #include
                                        buffer.push_str("\n", location);
                                        sz + 1
                                    }
                                    None => next.len(),
                                };
                                let remains = &next[..end].trim();
                                if remains.len() != 0 && !remains.starts_with("//") {
                                    return Err(PreprocessError::InvalidInclude);
                                }

                                let next = &next[end..];


                                Ok(next)
                            }
                            Err(()) => return Err(PreprocessError::FailedToFindFile),
                        }
                    }
                    None => return Err(PreprocessError::InvalidInclude),
                }
            }
            _ => return Err(PreprocessError::InvalidInclude),
        }
    } else if command.starts_with("ifdef") || command.starts_with("ifndef") {
        if skip {
            condition_chain.push(false);
            return Ok(get_normal_end(command));
        }
        let not = command.starts_with("ifndef");
        let next = if not {
            &command[6..]
        } else {
            &command[5..]
        };
        match next.chars().next() {
            Some(' ') | Some('\t') => {
                let args = next.trim_left();
                let end = match args.find('\n') {
                    Some(sz) => sz + 1,
                    _ => return Err(PreprocessError::InvalidIfndef(command.to_string())),
                };
                let body = &args[..end].trim();

                let exists = macros.iter().fold(false, |acc, m| acc || &m.0 == body);
                condition_chain.push(if not {
                    !exists
                } else {
                    exists
                });

                let remaining = &args[end..];
                Ok(remaining)
            }
            _ => return Err(PreprocessError::InvalidIfndef(command.to_string())),
        }
    } else if command.starts_with("if") {
        if skip {
            condition_chain.push(false);
            return Ok(get_normal_end(command));
        }
        let next = &command[2..];
        match next.chars().next() {
            Some(' ') | Some('\t') | Some('(') => {
                let args = next.trim_left();
                let end = match args.find('\n') {
                    Some(sz) => sz + 1,
                    _ => return Err(PreprocessError::InvalidIf(command.to_string())),
                };
                let body = &args[..end].trim();
                let remaining = &args[end..];

                let resolved = try!(SubstitutedText::new(body, StreamLocation(0))
                                        .apply_all(macros))
                                   .resolve();

                let resolved_str: &str = &resolved;
                // Sneaky hack to make `#if COND // comment` work
                let resolved_no_comment = match resolved_str.find("//") {
                    Some(sz) => resolved_str[..sz].trim(),
                    None => resolved_str,
                };

                let active = {
                    use condition_parser::parse;
                    try!(parse(resolved_no_comment))
                };
                condition_chain.push(active);

                Ok(remaining)
            }
            _ => return Err(PreprocessError::InvalidIf(command.to_string())),
        }
    } else if command.starts_with("else") {
        let next = &command[4..];
        match next.chars().next() {
            Some(' ') | Some('\t') | Some('\n') | Some('\r') => {
                let args = next;
                let end = match args.find('\n') {
                    Some(sz) => sz + 1,
                    _ => return Err(PreprocessError::InvalidElse),
                };
                let body = args[..end].trim();
                if body.len() != 0 && !body.starts_with("//") {
                    return Err(PreprocessError::InvalidElse);
                }

                try!(condition_chain.switch());

                let remaining = &args[end..];
                Ok(remaining)
            }
            _ => return Err(PreprocessError::InvalidElse),
        }
    } else if command.starts_with("endif") {
        let next = &command[5..];
        match next.chars().next() {
            Some(' ') | Some('\t') | Some('\n') | Some('\r') | None => {
                let args = next;
                let end = match args.find('\n') {
                    Some(sz) => sz + 1,
                    None => args.len(),
                };
                let body = &args[..end].trim();
                if body.len() != 0 && !body.starts_with("//") {
                    return Err(PreprocessError::InvalidEndIf);
                }

                try!(condition_chain.pop());

                let remaining = &args[end..];
                Ok(remaining)
            }
            _ => return Err(PreprocessError::InvalidEndIf),
        }
    } else if command.starts_with("define") {
        if skip {
            return Ok(get_macro_end(command));
        }
        let next = &command[6..];
        match next.chars().next() {
            Some(' ') | Some('\t') => {
                let mut remaining = next[1..].trim_left();

                // Consume define name
                let header_start = remaining;
                loop {
                    match remaining.chars().next() {
                        Some(c) if is_identifier_char(c) => {
                            remaining = &remaining[1..];
                        }
                        _ => break,
                    }
                }

                // Consume macro args
                match remaining.chars().next() {
                    Some('(') => {
                        remaining = &remaining[1..];
                        match remaining.find(')') {
                            Some(sz) => {
                                remaining = &remaining[(sz + 1)..];
                            }
                            None => return Err(PreprocessError::InvalidDefine),
                        }
                    }
                    _ => {}
                }

                // Let the header be the name + args
                let header = &header_start[..(header_start.len() - remaining.len())];

                // Consume gap between macro name/args and body
                let (body, remaining) = match remaining.chars().next() {
                    Some(' ') | Some('\t') | Some('\r') => {
                        remaining = &remaining[1..];
                        let sz = find_macro_end(remaining);
                        let body = &remaining[..sz];
                        (body, &remaining[(sz + 1)..])
                    }
                    Some('\n') => (&remaining[..1], &remaining[1..]),
                    None => ("", ""),
                    _ => return Err(PreprocessError::InvalidDefine),
                };

                let body = body.trim().replace("\\\n", "\n").replace("\\\r\n", "\r\n");
                let subbed_body = try!(SubstitutedText::new(&body, StreamLocation(0))
                                           .apply_all(&macros))
                                      .resolve();
                let macro_def = try!(Macro::from_definition(&header, &subbed_body, location));

                for current_macro in macros.iter() {
                    if *current_macro.0 == macro_def.0 {
                        return Err(PreprocessError::MacroAlreadyDefined(current_macro.0.clone()));
                    }
                }
                macros.push(macro_def);

                Ok(remaining)
            }
            _ => return Err(PreprocessError::InvalidDefine),
        }
    } else {
        return Err(PreprocessError::UnknownCommand);
    }
}

fn preprocess_file(buffer: &mut IntermediateText,
                   include_handler: &mut IncludeHandler,
                   file: &str,
                   macros: &mut Vec<Macro>,
                   condition_chain: &mut ConditionChain)
                   -> Result<(), PreprocessError> {

    let line_map = build_file_linemap(file, File::Unknown);
    let file_length = file.len() as u64;

    let mut stream = file;
    loop {
        let stream_location_in_file = StreamLocation(file_length - stream.len() as u64);
        let file_location = match line_map.get_file_location(&stream_location_in_file) {
            Ok(loc) => loc,
            Err(_) => panic!("could not find line for current position in file"),
        };
        let start_trimmed = stream.trim_left();
        if start_trimmed.starts_with("#") {
            let command = start_trimmed[1..].trim_left();
            stream = try!(preprocess_command(buffer,
                                             include_handler,
                                             command,
                                             file_location,
                                             macros,
                                             condition_chain));
        } else {

            fn find_region(mut stream: &str) -> (usize, bool) {
                let mut size = 0;
                let mut final_segment;
                loop {
                    let (sz, fs) = match stream.find('\n') {
                        Some(sz) => (sz + 1, false),
                        None => (stream.len(), true),
                    };
                    size = size + sz;
                    final_segment = fs;
                    stream = &stream[sz..];
                    if final_segment || stream.trim_left().starts_with("#") {
                        break;
                    }
                }
                (size, final_segment)
            }

            let (sz, final_segment) = find_region(stream);
            let line = &stream[..sz];
            stream = &stream[sz..];
            if condition_chain.is_active() {
                try!(SubstitutedText::new(line, stream_location_in_file).apply_all(macros))
                    .store(buffer, &line_map);
            }
            if final_segment {
                break;
            }
        }
    }

    Ok(())
}

pub fn preprocess(input: &str,
                  include_handler: &mut IncludeHandler)
                  -> Result<PreprocessedText, PreprocessError> {

    let mut intermediate_text = IntermediateText::new();
    let mut macros = vec![];
    let mut condition_chain = ConditionChain::new();
    try!(preprocess_file(&mut intermediate_text,
                         include_handler,
                         input,
                         &mut macros,
                         &mut condition_chain));

    if condition_chain.0.len() != 0 {
        return Err(PreprocessError::ConditionChainNotFinished);
    }

    Ok(PreprocessedText::from_intermediate_text(intermediate_text))
}

pub fn preprocess_single(input: &str) -> Result<PreprocessedText, PreprocessError> {
    preprocess(input, &mut NullIncludeHandler)
}

#[test]
fn test_empty() {
    assert_eq!(preprocess_single("").unwrap().code, b"");
    assert_eq!(preprocess_single("test").unwrap().code, b"test");
    assert_eq!(preprocess_single("t1\nt2").unwrap().code, b"t1\nt2");
    assert_eq!(preprocess_single("t1\r\nt2").unwrap().code, b"t1\r\nt2");
}

#[test]
fn test_define() {
    let pp = preprocess_single;
    assert_eq!(pp("#define X 0\nX").unwrap().code, b"0");
    assert_eq!(pp("#define X 0\nX X").unwrap().code, b"0 0");
    assert_eq!(pp("#define X 1\r\nX").unwrap().code, b"1");
    assert_eq!(pp("#define X 2\n#define Y X\nX").unwrap().code, b"2");
    assert_eq!(pp("#define X 2\\\n + 3\nX").unwrap().code, b"2\n + 3");
    assert_eq!(pp("#define X(a) a\nX(2)").unwrap().code, b"2");
    assert_eq!(pp("#define X(a,b) a+b\nX(2,3)").unwrap().code, b"2+3");
    assert_eq!(pp("#define X(X,b) X+b\nX(2,3)").unwrap().code, b"2+3");
    assert_eq!(pp("#define X(a,b) a+\\\nb\nX(2,3)").unwrap().code, b"2+\n3");
    assert_eq!(pp("#define X(a,b) a+\\\r\nb\nX(2,3)").unwrap().code,
               b"2+\r\n3");
    assert_eq!(pp("#define X").unwrap().code, b"");
    assert_eq!(pp("#define X 0\n#define Y 1\nX Y").unwrap().code, b"0 1");
    assert_eq!(pp("#define X 0\n#define XY 1\nXY X").unwrap().code, b"1 0");
    assert_eq!(pp("#define X(a) a\n#define Y 1\nX(Y)").unwrap().code, b"1");
    assert_eq!(pp("#define X(a,ab,ba,b) a ab a ba b ab a\nX(0,1,2,3)").unwrap().code,
               b"0 1 0 2 3 1 0");
}

#[test]
fn test_condition() {
    let pp = preprocess_single;
    assert!(pp("#if 0\nX").is_err());
    assert_eq!(pp("#if 0\nX\n#endif").unwrap().code, b"");
    assert_eq!(pp("#if 1\nX\n#endif").unwrap().code, b"X\n");
    assert_eq!(pp("#if 0\nX\n#else\nY\n#endif").unwrap().code, b"Y\n");
    assert_eq!(pp("#if 1\nX\n#else\nY\n#endif").unwrap().code, b"X\n");
    assert_eq!(pp("#if !0\nX\n#else\nY\n#endif").unwrap().code, b"X\n");
    assert_eq!(pp("#if !1\nX\n#else\nY\n#endif").unwrap().code, b"Y\n");
    assert_eq!(pp("#if\t 1  \n X  \n #else \n Y \n#endif \n\t").unwrap().code,
               b" X  \n\t");
    assert_eq!(pp("#define TRUE 1\n#if TRUE\nX\n#else\nY\n#endif").unwrap().code,
               b"X\n");
    assert_eq!(pp("#define TRUE\n#ifdef TRUE\nX\n#else\nY\n#endif").unwrap().code,
               b"X\n");
    assert_eq!(pp("#define TRUE\n#ifndef TRUE\nX\n#else\nY\n#endif").unwrap().code,
               b"Y\n");
    assert_eq!(pp("#define TRUE 1\n#ifdef TRUE\nX\n#else\nY\n#endif").unwrap().code,
               b"X\n");
    assert_eq!(pp("#define TRUE 0\n#ifndef TRUE\nX\n#else\nY\n#endif").unwrap().code,
               b"Y\n");
    assert_eq!(pp("#if 0\n#define X Y\n#endif\nX").unwrap().code, b"X");
    assert_eq!(pp("#if 1\n#define X Y\n#else\n#define X Z\n#endif\nX").unwrap().code,
               b"Y");
    assert_eq!(pp("#if 1\n#define X Y\n#else\n#include\"fail\"\n#endif\nX").unwrap().code,
               b"Y");
    assert_eq!(pp("#if 1 // comment\n#define X Y\n#else // comment\n#include\"fail\"\n#endif // \
                   comment\nX")
                   .unwrap()
                   .code,
               b"Y");
}

#[test]
fn test_include() {
    struct TestFileLoader;
    impl IncludeHandler for TestFileLoader {
        fn load(&self, file_name: &str) -> Result<String, ()> {
            Ok(match file_name.as_ref() {
                   "1.csh" => "X",
                   "2.csh" => "Y",
                   _ => return Err(()),
               }
               .to_string())
        }
    }

    fn pf(contents: &str) -> Result<PreprocessedText, PreprocessError> {
        preprocess(contents, &TestFileLoader)
    }

    // Unknown files should always fail
    assert!(pf("#include \"unknown.csh\"").is_err());
    assert!(pf("#include").is_err());
    assert!(pf("#include\n").is_err());
    // Normal case
    assert_eq!(pf("#include \"1.csh\"\n").unwrap().code, b"X\n");
    // End of file include
    assert_eq!(pf("#include \"1.csh\"").unwrap().code, b"X");
    // Extra whitespace
    assert_eq!(pf("#include \"1.csh\"\t\n").unwrap().code, b"X\n");
    // Less whitespace
    assert_eq!(pf("#include\"1.csh\"\n").unwrap().code, b"X\n");
    // Alternative delimiters (not treated differently currently)
    assert_eq!(pf("#include <1.csh>\n").unwrap().code, b"X\n");
    assert_eq!(pf("#include<1.csh>\n").unwrap().code, b"X\n");
    assert!(pf("#include \"1.csh>\n").is_err());
    assert!(pf("#include <1.csh\"\n").is_err());
    // Comments after includes needs to work
    assert_eq!(pf("#include \"1.csh\" // include \n").unwrap().code, b"X\n");
    assert_eq!(pf("#include \"1.csh\"\n#include \"2.csh\"").unwrap().code,
               b"X\nY");
    // We don't want to read files that are #if'd out
    assert_eq!(pf("#if 1\n#include \"1.csh\"\n#else\n#include \"unknown.csh\"\n#endif")
                   .unwrap()
                   .code,
               b"X\n");
    assert_eq!(pf("#if 0\n#include \"unknown.csh\"\n#else\n#include \"2.csh\"\n#endif")
                   .unwrap()
                   .code,
               b"Y\n");
}
