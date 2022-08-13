use std::collections::HashMap;

/// A file used as an input
#[derive(PartialEq, Debug, Clone)]
pub struct FileName(pub String);

/// A line number in a file
#[derive(PartialEq, Debug, Clone)]
pub struct Line(pub u64);

/// The column index in a line
#[derive(PartialEq, Debug, Clone)]
pub struct Column(pub u64);

/// Fully qualified location
#[derive(PartialEq, Debug, Clone)]
pub struct FileLocation(
    // TODO: Avoid using a string here so this can be used where it is replicated many times
    pub FileName,
    pub Line,
    pub Column,
);

impl FileLocation {
    pub fn none() -> FileLocation {
        FileLocation(FileName(String::new()), Line(0), Column(0))
    }
}

/// The raw number of bytes from the start of a stream
#[derive(PartialEq, Debug, Clone)]
pub struct StreamLocation(pub u64);

/// Wrapper to pair a node with a FileLocation
#[derive(PartialEq, Debug, Clone)]
pub struct Located<T> {
    pub node: T,
    pub location: FileLocation,
}

impl<T> Located<T> {
    pub fn new(node: T, loc: FileLocation) -> Located<T> {
        Located {
            node: node,
            location: loc,
        }
    }
    pub fn to_node(self) -> T {
        self.node
    }
    pub fn to_loc(self) -> FileLocation {
        self.location
    }
    pub fn loc(line: u64, column: u64, node: T) -> Located<T> {
        Located {
            node: node,
            location: FileLocation(FileName(String::new()), Line(line), Column(column)),
        }
    }
    pub fn none(node: T) -> Located<T> {
        Located {
            node: node,
            location: FileLocation::none(),
        }
    }
}

impl<T> std::ops::Deref for Located<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.node
    }
}

/// Error cases for file loading
#[derive(PartialEq, Debug, Clone)]
pub enum IncludeError {
    FileNotFound,
    FileNotText,
}

/// Trait for loading files from #include directives
pub trait IncludeHandler {
    fn load(&mut self, file_name: &str) -> Result<String, IncludeError>;
}

/// A file loader that fails to load any files
pub struct NullIncludeHandler;

impl IncludeHandler for NullIncludeHandler {
    fn load(&mut self, _: &str) -> Result<String, IncludeError> {
        Err(IncludeError::FileNotFound)
    }
}

pub type KernelParamSlot = u32;

#[derive(PartialEq, Debug, Clone)]
pub struct BindMap {
    pub read_map: HashMap<u32, KernelParamSlot>,
    pub write_map: HashMap<u32, KernelParamSlot>,
    pub cbuffer_map: HashMap<u32, KernelParamSlot>,
    pub sampler_map: HashMap<u32, KernelParamSlot>,
}

impl BindMap {
    pub fn new() -> BindMap {
        BindMap {
            read_map: HashMap::new(),
            write_map: HashMap::new(),
            cbuffer_map: HashMap::new(),
            sampler_map: HashMap::new(),
        }
    }
}

pub mod opencl;
