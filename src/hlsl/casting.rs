
use super::ir::*;

// Overload priority
// =================
//
// Testing the priority of functions when passed arguments of a given type:
//
// bool (+ bool literals):           bool     -> uint/int/float/double                              -> half
// int (+ type casted int literals): int      -> uint                  -> bool -> float/double      -> half
// untyped int literal:                          uint/int              -> bool -> float/double/half
// uint (+ uint literals):           uint     -> int                   -> bool -> float/double      -> half
// half:                             half     -> float/double                  -> bool/int/uint
// float:                            float    -> double                        -> bool/int/uint/half
// double:                           double                                    -> bool/int/uint/float/half
//
// lvalue / rvalue didn't seem to make a difference. These don't seem consistent with C or C++
// rules.
//
// Priority:
// 1) Exact matches fit first (untyped int literal could be either)
// 2) Promotion
// 3) Bool if the source is an int
// 4) Any convertible type, except:
// 5) Halfs if you're an bool/int/uint, unless it's an untyped literal (something to do with halfs being smaller than int/uint but not literals???)
//
// I was going to try to implement nice logic in here, but I honestly have no idea what rules govern these priorities.

#[derive(PartialEq, Debug, Clone)]
pub enum ConversionRank {
    Exact,
    Promotion,
    IntToBool,
    Conversion,
    HalfIsASecondClassCitizen,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ConversionPriority {
    Better,
    Equal,
    Worse,
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum NumericDimension {
    Scalar,
    Vector(u32),
    Matrix(u32, u32),
}

#[derive(PartialEq, Debug, Clone)]
struct NumericCast(ScalarType, ScalarType, NumericDimension);

#[derive(PartialEq, Debug, Clone)]
pub struct ImplicitConversion(Type, Option<NumericCast>);

impl ConversionRank {
    pub fn compare(&self, other: &ConversionRank) -> ConversionPriority {
        let my_order = self.order();
        let other_order = other.order();
        match (my_order < other_order, my_order <= other_order) {
            (false, false) => ConversionPriority::Worse,
            (false, true) => ConversionPriority::Equal,
            (true, false) => unreachable!(),
            (true, true) => ConversionPriority::Better,
        }
    }

    fn order(&self) -> u32 {
        match *self {
            ConversionRank::Exact => 0,
            ConversionRank::Promotion => 1,
            ConversionRank::IntToBool => 2,
            ConversionRank::Conversion => 3,
            ConversionRank::HalfIsASecondClassCitizen => 4,
        }
    }
}


impl NumericCast {

    fn new(source: &ScalarType, dest: &ScalarType, dim: NumericDimension) -> Result<NumericCast, ()> {
        match *dest {
            _ if source == dest => Err(()),
            ScalarType::Bool | ScalarType::Int | ScalarType::UInt | ScalarType::Half | ScalarType::Float | ScalarType::Double => {
                Ok(NumericCast(source.clone(), dest.clone(), dim))
            },
            ScalarType::UntypedInt => Err(()),
        }
    }

    fn find_all(source: &ScalarType, dim: NumericDimension) -> Vec<NumericCast> {
        let mut vec = vec![];
        for dest in &[ScalarType::Bool, ScalarType::Int, ScalarType::UInt, ScalarType::Half, ScalarType::Float, ScalarType::Double] {
            if let Ok(cast) = NumericCast::new(source, dest, dim.clone()) {
                vec.push(cast);
            }
        };
        vec
    }

    fn get_rank(&self) -> ConversionRank {
        match *self {
            NumericCast(ScalarType::Bool, ref dest, _) => {
                match *dest {
                    ScalarType::Bool => ConversionRank::Exact,
                    ScalarType::Int | ScalarType::UInt | ScalarType::Float | ScalarType::Double => ConversionRank::Conversion,
                    ScalarType::Half => ConversionRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::UntypedInt, ref dest, _) => {
                match *dest {
                    ScalarType::Int | ScalarType::UInt => ConversionRank::Promotion,
                    ScalarType::Bool => ConversionRank::IntToBool,
                    ScalarType::Float | ScalarType::Double | ScalarType::Half => ConversionRank::Conversion,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::Int, ref dest, _) => {
                match *dest {
                    ScalarType::Int => ConversionRank::Exact,
                    ScalarType::UInt => ConversionRank::Promotion,
                    ScalarType::Bool => ConversionRank::IntToBool,
                    ScalarType::Float | ScalarType::Double => ConversionRank::Conversion,
                    ScalarType::Half => ConversionRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::UInt, ref dest, _) => {
                match *dest {
                    ScalarType::UInt => ConversionRank::Exact,
                    ScalarType::Int => ConversionRank::Promotion,
                    ScalarType::Bool => ConversionRank::IntToBool,
                    ScalarType::Float | ScalarType::Double => ConversionRank::Conversion,
                    ScalarType::Half => ConversionRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::Half, ref dest, _) => {
                match *dest {
                    ScalarType::Half => ConversionRank::Exact,
                    ScalarType::Float | ScalarType::Double => ConversionRank::Promotion,
                    ScalarType::Bool | ScalarType::Int | ScalarType::UInt => ConversionRank::Conversion,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::Float, ref dest, _) => {
                match *dest {
                    ScalarType::Float => ConversionRank::Exact,
                    ScalarType::Double => ConversionRank::Promotion,
                    ScalarType::Bool | ScalarType::Int | ScalarType::UInt | ScalarType::Half => ConversionRank::Conversion,
                    ScalarType::UntypedInt => panic!(),
                }
            },
            NumericCast(ScalarType::Double, ref dest, _) => {
                match *dest {
                    ScalarType::Double => ConversionRank::Exact,
                    ScalarType::Bool | ScalarType::Int | ScalarType::UInt | ScalarType::Half | ScalarType::Float => ConversionRank::Conversion,
                    ScalarType::UntypedInt => panic!(),
                }
            },
        }
    }

    fn get_target_type(&self) -> Type {
        Type::Structured(StructuredType::Data(match self.2 {
            NumericDimension::Scalar => DataType::Scalar(self.1.clone()),
            NumericDimension::Vector(ref x) => DataType::Vector(self.1.clone(), *x),
            NumericDimension::Matrix(ref x, ref y) => DataType::Matrix(self.1.clone(), *x, *y),
        }))
    }

    fn apply(&self, expr: Expression) -> Expression {
        let &NumericCast(_, ref dest, ref dim) = self;
        let to_type_data = match *dim {
            NumericDimension::Scalar => DataType::Scalar(dest.clone()),
            NumericDimension::Vector(ref x) => DataType::Vector(dest.clone(), *x),
            NumericDimension::Matrix(ref x, ref y) => DataType::Matrix(dest.clone(), *x, *y),
        };
        Expression::Cast(Type::Structured(StructuredType::Data(to_type_data)), Box::new(expr))
    }
}

impl ImplicitConversion {

    pub fn find(source: &Type, dest: &Type) -> Result<ImplicitConversion, ()> {
        match (source, dest) {
            (ref ty1, ref ty2) if ty1 == ty2 => Ok(ImplicitConversion(source.clone(), None)),
            (&Type::Structured(StructuredType::Data(ref d1)), &Type::Structured(StructuredType::Data(ref d2))) => ImplicitConversion::find_data(d1, d2),
            // Struct casts only supported for same type structs
            _ => Err(()),
        }
    }

    fn find_data(source: &DataType, dest: &DataType) -> Result<ImplicitConversion, ()> {
        let full_type = Type::Structured(StructuredType::Data(source.clone()));
        match (source, dest) {
            (ref ty1, ref ty2) if ty1 == ty2 => Ok(ImplicitConversion(full_type, None)),
            (&DataType::Scalar(ref s1), &DataType::Scalar(ref s2)) => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Scalar));
                Ok(ImplicitConversion(full_type, Some(cast)))
            },
            (&DataType::Vector(ref s1, ref x1), &DataType::Vector(ref s2, ref x2)) if x1 == x2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Vector(*x2)));
                Ok(ImplicitConversion(full_type, Some(cast)))
            },
            (&DataType::Matrix(ref s1, ref x1, ref y1), &DataType::Matrix(ref s2, ref x2, ref y2)) if x1 == x2 && y1 == y2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Matrix(*x2, *y2)));
                Ok(ImplicitConversion(full_type, Some(cast)))
            },
            // Vector <-> Matrix casts not implemented
            _ => Err(()),
        }
    }

    pub fn find_all(source: &Type) -> Vec<ImplicitConversion> {
        let mut vec = vec![ImplicitConversion(source.clone(), None)];
        match *source {
            Type::Structured(StructuredType::Data(DataType::Scalar(ref scalar))) => {
                for nc in NumericCast::find_all(scalar, NumericDimension::Scalar) {
                    vec.push(ImplicitConversion(source.clone(), Some(nc)));
                };
            },
            Type::Structured(StructuredType::Data(DataType::Vector(ref scalar, ref x))) => {
                for nc in NumericCast::find_all(scalar, NumericDimension::Vector(*x)) {
                    vec.push(ImplicitConversion(source.clone(), Some(nc)));
                };
            },
            Type::Structured(StructuredType::Data(DataType::Matrix(ref scalar, ref x, ref y))) => {
                for nc in NumericCast::find_all(scalar, NumericDimension::Matrix(*x, *y)) {
                    vec.push(ImplicitConversion(source.clone(), Some(nc)));
                };
            },
            _ => { },
        };
        vec
    }

    pub fn get_rank(&self) -> ConversionRank {
        match *self {
            ImplicitConversion(_, None) => ConversionRank::Exact,
            ImplicitConversion(_, Some(ref numeric)) => numeric.get_rank(),
        }
    }

     pub fn get_target_type(&self) -> Type {
        match *self {
            ImplicitConversion(ref source_type, None) => source_type.clone(),
            ImplicitConversion(_, Some(ref numeric)) => numeric.get_target_type(),
        }
    }

    pub fn apply(&self, expr: Expression) -> Expression {
        match *self {
            ImplicitConversion(_, None) => expr,
            ImplicitConversion(_, Some(ref cast)) => cast.apply(expr),
        }
    }
}
