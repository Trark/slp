
use std::error;
use std::fmt;
use StreamLocation;
use FileLocation;
use File;
use Line;
use Column;
use FileLoader;

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
    pub fn as_bytes(&self) -> &[u8] {
        &self.code
    }
    pub fn get_file_location(&self, stream_location: &StreamLocation) -> Result<FileLocation, ()> {
        self.debug_locations.get_file_location(stream_location)
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

fn preprocess_file(buffer: String, file_loader: &FileLoader, file: &str, debug_locations: &mut Vec<(StreamLocation, FileLocation)>) -> Result<String, PreprocessError> {
    let lines = file.split('\n').collect::<Vec<_>>();
    let mut buffer = buffer;
    for (line_index, line) in lines.iter().enumerate() {

        let trimmed = line.trim_left();
        if trimmed.starts_with("#") {
            let trimmed = trimmed[1..].trim_left();
            if trimmed.starts_with("include") {
                let next = &trimmed[7..];
                match next.chars().next() {
                    Some(' ') | Some('\t') | Some('"') | Some('<') => {
                        let next = next.trim_left();
                        let end = match next.chars().next() {
                            Some('"') => '"',
                            Some('<') => '>',
                            _ => return Err(PreprocessError::InvalidInclude),
                        };
                        let next = &next[1..];
                        match next.find(end) {
                            Some(sz) => {
                                let file_name = &next[..sz];
                                // Ignore the rest of the line

                                // Include the file
                                match file_loader.load(file_name) {
                                    Ok(file) => {
                                        buffer = try!(preprocess_file(buffer, file_loader, &file, debug_locations));
                                        // Push a new line so the last line of the include file is on a
                                        // separate line to the first line after the #include
                                        buffer.push('\n');
                                    },
                                    Err(()) => return Err(PreprocessError::FailedToFindFile),
                                }
                            }
                            None => return Err(PreprocessError::InvalidInclude),
                        }
                    }
                    _ => return Err(PreprocessError::InvalidInclude),
                }
            } else {
                return Err(PreprocessError::UnknownCommand);
            }

            // Finish processing the current line
            continue;
        }

        // Add the current location to the line map
        let stream_offset = buffer.as_bytes().len() as u64;
        let line_number = (line_index + 1) as u64;
        let column_number = 1; // The first index will display as column 1
        debug_locations.push((
            StreamLocation(stream_offset),
            FileLocation(File::Unknown, Line(line_number), Column(column_number))
        ));

        // Add line to the preprocessed text buffer
        buffer.push_str(line);
        if line_index != lines.len() - 1 {
            // Add the line end that was removed by the split if we're
            // not on the last line
            buffer.push('\n');
        }
    }
    Ok(buffer)
}

pub fn preprocess(input: &str, file_loader: &FileLoader) -> Result<PreprocessedText, PreprocessError> {

    let mut lines = vec![];
    let code = try!(preprocess_file(String::new(), file_loader, input, &mut lines));

    Ok(PreprocessedText {
        code: code.into_bytes(),
        debug_locations: LineMap { lines: lines },
    })
}

pub fn preprocess_single(input: &str) -> Result<PreprocessedText, PreprocessError> {
    use NullFileLoader;
    preprocess(input, &NullFileLoader)
}
