
use std::error;
use std::fmt;
use StreamLocation;
use FileLocation;
use File;
use Line;
use Column;
use IncludeHandler;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessError {
    UnknownCommand,
    InvalidInclude,
    FailedToFindFile,
}

impl error::Error for PreprocessError {
    fn description(&self) -> &str {
        match *self {
            PreprocessError::UnknownCommand => "unknown preprocessor command",
            PreprocessError::InvalidInclude => "invalid #include command",
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
    fn new() -> IntermediateText { IntermediateText { buffer: String::new(), debug_locations: LineMap { lines: vec![] } } }
    fn push_str(&mut self, segment: &str, segment_location: FileLocation) {
        match segment.find('\n') {
            Some(sz) => assert_eq!(sz, segment.len() - 1),
            None => { },
        }
        let stream_location_in_buffer = StreamLocation(self.buffer.len() as u64);
        self.buffer.push_str(segment);
        self.debug_locations.lines.push((stream_location_in_buffer, segment_location));
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
        line_map.lines.push((StreamLocation(file_length - length_left), FileLocation(file_name.clone(), Line(current_line), Column(1))));
        current_line = current_line + 1;
        stream = &stream[sz..];
        if final_segment {
            break;
        }
    }
    line_map
}

fn preprocess_command<'a>(buffer: &mut IntermediateText, include_handler: &IncludeHandler, command: &'a str, location: FileLocation) -> Result<&'a str, PreprocessError> {
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
                                try!(preprocess_file(buffer, include_handler, &file));
                                // Push a new line so the last line of the include file is on a
                                // separate line to the first line after the #include
                                buffer.push_str("\n", location);

                                Ok(&args[(sz + 1)..])
                            },
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
        panic!("unimplemented: #define")
    } else {
        return Err(PreprocessError::UnknownCommand);
    }
}

fn preprocess_file(buffer: &mut IntermediateText, include_handler: &IncludeHandler, file: &str) -> Result<(), PreprocessError> {

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
            stream = try!(preprocess_command(buffer, include_handler, command, file_location));
        } else {
            // Normal line
            let (sz, final_segment) = match stream.find('\n') {
                Some(sz) => (sz + 1, false),
                None => (stream.len(), true),
            };
            let line = &stream[..sz];
            buffer.push_str(line, file_location);
            stream = &stream[sz..];
            if final_segment {
                break;
            }
        }
    }

    Ok(())
}

pub fn preprocess(input: &str, include_handler: &IncludeHandler) -> Result<PreprocessedText, PreprocessError> {

    let mut intermediate_text = IntermediateText::new();
    try!(preprocess_file(&mut intermediate_text, include_handler, input));

    Ok(PreprocessedText::from_intermediate_text(intermediate_text))
}

pub fn preprocess_single(input: &str) -> Result<PreprocessedText, PreprocessError> {
    use NullIncludeHandler;
    preprocess(input, &NullIncludeHandler)
}
