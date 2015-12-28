
use std::error;
use std::fmt;
use slp_shared::*;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessError {
    UnknownCommand,
    InvalidInclude,
    InvalidDefine,
    MacroRequiresArguments,
    MacroArgumentsNeverEnd,
    MacroExpectsDifferentNumberOfArguments,
    FailedToFindFile,
}

impl error::Error for PreprocessError {
    fn description(&self) -> &str {
        match *self {
            PreprocessError::UnknownCommand => "unknown preprocessor command",
            PreprocessError::InvalidInclude => "invalid #include command",
            PreprocessError::InvalidDefine => "invalid #define command",
            PreprocessError::MacroRequiresArguments => "macro function requires arguments",
            PreprocessError::MacroArgumentsNeverEnd => "expected end of macro arguments",
            PreprocessError::MacroExpectsDifferentNumberOfArguments => {
                "macro requires different number of arguments"
            }
            PreprocessError::FailedToFindFile => "could not find file",
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
        let mut last_line = None;
        for (line_index, &(ref line_stream, _)) in self.lines.iter().enumerate() {
            if line_stream.0 <= stream_location.0 {
                last_line = Some(line_index);
            }
        }
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
                match text.find(arg) {
                    Some(sz) => {
                        let before = &text[..sz];
                        let after_offset = sz + arg.len();
                        let after = &text[after_offset..];
                        let separated_before = match before.chars().last() {
                            Some(c) => is_identifier_char(c),
                            None => false,
                        };
                        let separated_after = match after.chars().next() {
                            Some(c) => is_identifier_char(c),
                            None => false,
                        };
                        if !separated_before && !separated_after {
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

impl SubstitutedSegment {
    fn apply(self,
             macro_def: &Macro,
             output: &mut Vec<SubstitutedSegment>)
             -> Result<(), PreprocessError> {
        match self {
            SubstitutedSegment::Text(text, location) => {
                match text.find(&macro_def.0) {
                    Some(sz) => {
                        let before = &text[..sz];
                        let after_offset = sz + macro_def.0.len();
                        let mut remaining = &text[after_offset..];

                        let separated_before = match before.chars().last() {
                            Some(c) => is_identifier_char(c),
                            None => false,
                        };
                        let separated_after = match remaining.chars().next() {
                            Some(c) => is_identifier_char(c),
                            None => false,
                        };
                        if !separated_before && !separated_after {

                            // Read macro arguments
                            let arguments = if macro_def.1 > 0 {
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
                                    let (sz, last) = match (remaining.find(','),
                                                            remaining.find(")")) {
                                        (Some(szn), Some(szl)) if szn < szl => (szn, false),
                                        (_, Some(szl)) => (szl, true),
                                        (Some(szn), None) => (szn, false),
                                        (None, None) => {
                                            return Err(PreprocessError::MacroArgumentsNeverEnd)
                                        }
                                    };
                                    let arg = remaining[..sz].trim();
                                    remaining = &remaining[(sz + 1)..];
                                    args.push(arg);
                                    if last {
                                        break;
                                    }
                                }

                                args
                            } else {
                                vec![]
                            };
                            let after = remaining;

                            if arguments.len() as u64 != macro_def.1 {
                                return Err(PreprocessError::MacroExpectsDifferentNumberOfArguments);
                            }

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
                                        replaced_text.push_str(arguments[*index as usize])
                                    }
                                }
                            }
                            if replaced_text.len() > 0 {
                                output.push(SubstitutedSegment::Replaced(replaced_text,
                                                                         macro_def.3.clone()));
                            }
                            if after.len() > 0 {
                                try!(SubstitutedSegment::Text(after.to_string(), after_location)
                                         .apply(macro_def, output));
                            }
                            return Ok(());
                        }
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

    fn apply_all(self, macros_defs: &[Macro]) -> Result<SubstitutedText, PreprocessError> {
        let length = self.0.len();
        Ok(SubstitutedText(try!(self.0
                                    .into_iter()
                                    .fold(Ok(Vec::with_capacity(length)), |vec_res, segment| {
                                        let mut vec = try!(vec_res);
                                        let mut last_segments = vec![segment];
                                        for macro_def in macros_defs {
                                            let mut next_segments =
                                                Vec::with_capacity(last_segments.len());
                                            for substituted_segment in last_segments {
                                                try!(substituted_segment.apply(macro_def,
                                                                               &mut next_segments));
                                            }
                                            last_segments = next_segments;
                                        }
                                        vec.append(&mut last_segments);
                                        Ok(vec)
                                    }))))
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

fn preprocess_command<'a>(buffer: &mut IntermediateText,
                          include_handler: &IncludeHandler,
                          command: &'a str,
                          location: FileLocation,
                          macros: &mut Vec<Macro>)
                          -> Result<&'a str, PreprocessError> {
    if command.starts_with("include") {
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
                                try!(preprocess_file(buffer, include_handler, &file, macros));
                                // Push a new line so the last line of the include file is on a
                                // separate line to the first line after the #include
                                buffer.push_str("\n", location);

                                Ok(&args[(sz + 1)..])
                            }
                            Err(()) => return Err(PreprocessError::FailedToFindFile),
                        }
                    }
                    None => return Err(PreprocessError::InvalidInclude),
                }
            }
            _ => return Err(PreprocessError::InvalidInclude),
        }
    } else if command.starts_with("if") {
        panic!("unimplemented: #if")
    } else if command.starts_with("else") {
        panic!("unimplemented: #else")
    } else if command.starts_with("endif") {
        panic!("unimplemented: #endif")
    } else if command.starts_with("define") {
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
                match remaining.chars().next() {
                    Some(' ') | Some('\t') => {
                        remaining = &remaining[1..];
                    }
                    _ => return Err(PreprocessError::InvalidDefine),
                };

                // Consume up to first real char in body
                remaining = remaining.trim_left();

                fn find_macro_end(mut remaining: &str) -> Option<usize> {
                    let initial_length = remaining.len();
                    loop {
                        match remaining.find('\n') {
                            Some(sz) => {
                                let before = &remaining[..(sz - 1)];
                                remaining = &remaining[(sz + 1)..];
                                match before.chars().last() {
                                    Some(x) if x == '\\' => {}
                                    _ => break,
                                }
                            }
                            _ => return None,
                        }
                    }
                    Some(initial_length - remaining.len() - 1)
                }

                match find_macro_end(remaining) {
                    Some(sz) => {
                        let body = &remaining[..sz]
                                        .trim()
                                        .replace("\\\n", "\n")
                                        .replace("\\\r\n", "\n");
                        let subbed_body = try!(SubstitutedText::new(body, StreamLocation(0))
                                                   .apply_all(&macros))
                                              .resolve();
                        let macro_def = try!(Macro::from_definition(&header,
                                                                    &subbed_body,
                                                                    location));

                        for current_macro in macros.iter() {
                            if *current_macro.0 == macro_def.0 {
                                return Err(PreprocessError::InvalidDefine);
                            }
                        }
                        macros.push(macro_def);

                        Ok(&remaining[(sz + 1)..])
                    }
                    None => return Err(PreprocessError::InvalidDefine),
                }
            }
            _ => return Err(PreprocessError::InvalidDefine),
        }
    } else {
        return Err(PreprocessError::UnknownCommand);
    }
}

fn preprocess_file(buffer: &mut IntermediateText,
                   include_handler: &IncludeHandler,
                   file: &str,
                   macros: &mut Vec<Macro>)
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
                                             macros));
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
            try!(SubstitutedText::new(line, stream_location_in_file).apply_all(macros))
                .store(buffer, &line_map);
            if final_segment {
                break;
            }
        }
    }

    Ok(())
}

pub fn preprocess(input: &str,
                  include_handler: &IncludeHandler)
                  -> Result<PreprocessedText, PreprocessError> {

    let mut intermediate_text = IntermediateText::new();
    let mut macros = vec![];
    try!(preprocess_file(&mut intermediate_text, include_handler, input, &mut macros));

    Ok(PreprocessedText::from_intermediate_text(intermediate_text))
}

pub fn preprocess_single(input: &str) -> Result<PreprocessedText, PreprocessError> {
    preprocess(input, &NullIncludeHandler)
}
