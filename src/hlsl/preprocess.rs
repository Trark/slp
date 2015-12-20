
use std::error;
use std::fmt;
use StreamLocation;
use FileLocation;
use File;
use Line;
use Column;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessError {
    
}

impl error::Error for PreprocessError {
    fn description(&self) -> &str {
        match *self {
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

fn preprocess_file(buffer: String, file: &str, debug_locations: &mut Vec<(StreamLocation, FileLocation)>) -> String {
    let lines = file.split('\n').collect::<Vec<_>>();
    let mut buffer = buffer;
    for (line_index, line) in lines.iter().enumerate() {

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
    buffer
}

pub fn preprocess(input: &str) -> Result<PreprocessedText, PreprocessError> {

    let mut lines = vec![];
    let code = preprocess_file(String::new(), input, &mut lines);

    Ok(PreprocessedText {
        code: code.into_bytes(),
        debug_locations: LineMap { lines: lines },
    })
}
