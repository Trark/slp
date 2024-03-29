use docopt::Docopt;
use serde::Deserialize;
use slp_sequence_hlsl_to_cl::*;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::thread;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Slipstream HLSL to OpenCL Compiler

Usage:
  slipstream [options] [-I <include_path>...] <source-file>
  slipstream --help

Options:
  -h --help                    Show help.
  --entry-point <entry_point>  Entry function [default: CSMAIN].
  -o <output_file>             Output file.
  -I <include_path>            Path to search for includes in.
";

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct Args {
    flag_entry_point: String,
    flag_o: Option<String>,
    flag_I: Vec<String>,
    arg_source_file: String,
}

struct FileLoader {
    include_paths: Vec<PathBuf>,
}

impl IncludeHandler for FileLoader {
    fn load(&mut self, file_name: &str) -> Result<String, IncludeError> {
        let file_path = Path::new(file_name);
        let mut file_res = File::open(file_path);
        if file_path.is_relative() {
            for path in &self.include_paths {
                match file_res {
                    Ok(_) => {}
                    Err(_) => {
                        let mut p = path.clone();
                        p.push(file_path);
                        file_res = File::open(p);
                    }
                }
            }
        }
        match file_res {
            Ok(mut file) => {
                let mut buffer = String::new();
                match file.read_to_string(&mut buffer) {
                    Ok(_) => Ok(buffer),
                    Err(_) => Err(IncludeError::FileNotText),
                }
            }
            Err(_) => Err(IncludeError::FileNotFound),
        }
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let Args {
        flag_entry_point,
        flag_o: flag_output_file,
        flag_I: flag_include_paths,
        arg_source_file,
    } = args;

    let mut source_file = match File::open(&arg_source_file) {
        Ok(file) => file,
        Err(_) => {
            println!("Failed to load file '{}'", &arg_source_file);
            return;
        }
    };

    let mut source_contents = String::new();
    match source_file.read_to_string(&mut source_contents) {
        Ok(_) => {}
        Err(_) => {
            println!("Failed to read file '{}'", &arg_source_file);
            return;
        }
    };

    let paths = flag_include_paths
        .iter()
        .map(|s| PathBuf::from(s))
        .collect::<Vec<_>>();
    for path in &paths {
        if !path.is_dir() {
            let s = match path.to_str() {
                Some(s) => s,
                None => "<invalid>",
            };
            println!("Include path '{}' is not a directory", s);
            return;
        }
    }

    let join_handle_res = thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let include_handler = Box::new(FileLoader {
                include_paths: paths,
            });
            let entry_point = flag_entry_point.clone();
            let kernel_name = flag_entry_point;

            let input = Input {
                entry_point: entry_point,
                main_file: source_contents,
                main_file_name: arg_source_file,
                file_loader: include_handler,
                kernel_name: kernel_name,
            };

            hlsl_to_cl(input)
        });

    let join_handle = match join_handle_res {
        Ok(jh) => jh,
        Err(_) => {
            println!("Failed to start worker thread");
            return;
        }
    };

    let output_res = match join_handle.join() {
        Ok(output) => output,
        Err(_) => {
            println!("Failed to join worker thread");
            return;
        }
    };

    match output_res {
        Ok(output) => match flag_output_file {
            Some(output_file) => {
                let mut file = match File::create(&output_file) {
                    Ok(file) => file,
                    Err(_) => {
                        println!("Failed to open output file {}", output_file);
                        return;
                    }
                };
                match file.write(output.code.to_string().as_bytes()) {
                    Ok(_) => {}
                    Err(_) => println!("Failed to write to output file"),
                };
            }
            None => println!("{}", output.code.to_string()),
        },
        Err(err) => println!("Error: {:?}", err),
    };
}
