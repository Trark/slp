
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
struct ValueTypeCast(Type, ValueType, ValueType);

#[derive(PartialEq, Debug, Clone)]
pub struct ImplicitConversion(ExpressionType, Option<ValueTypeCast>, Option<NumericCast>);

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

    fn get_target_type(&self) -> ExpressionType {
        Type::from_layout(match self.2 {
            NumericDimension::Scalar => TypeLayout::Scalar(self.1.clone()),
            NumericDimension::Vector(ref x) => TypeLayout::Vector(self.1.clone(), *x),
            NumericDimension::Matrix(ref x, ref y) => TypeLayout::Matrix(self.1.clone(), *x, *y),
        }).to_rvalue()
    }

    fn apply(&self, expr: Expression) -> Expression {
        let &NumericCast(_, ref dest, ref dim) = self;
        let to_type_data = match *dim {
            NumericDimension::Scalar => TypeLayout::Scalar(dest.clone()),
            NumericDimension::Vector(ref x) => TypeLayout::Vector(dest.clone(), *x),
            NumericDimension::Matrix(ref x, ref y) => TypeLayout::Matrix(dest.clone(), *x, *y),
        };
        Expression::Cast(Type::from_layout(to_type_data), Box::new(expr))
    }
}

impl ValueTypeCast {
    fn get_target_type(&self) -> ExpressionType {
        ExpressionType(self.0.clone(), self.2.clone())
    }
}

impl ImplicitConversion {

    pub fn find(source: &ExpressionType, dest: &ExpressionType) -> Result<ImplicitConversion, ()> {

        let source_copy = source.clone();
        let (source_type, dest_type, value_type_cast) = match (&source.1, &dest.1) {
            (&ValueType::Rvalue, &ValueType::Lvalue) => return Err(()),
            (&ValueType::Rvalue, &ValueType::Rvalue) | (&ValueType::Lvalue, &ValueType::Lvalue) => (&source.0, &dest.0, None),
            (&ValueType::Lvalue, &ValueType::Rvalue) => (&source.0, &dest.0, Some(ValueTypeCast(source.0.clone(), ValueType::Lvalue, ValueType::Rvalue))),
        };

        let &Type(ref source_l, ref mods) = source_type;
        assert!(mods.is_empty());
        let &Type(ref dest_l, ref modd) = dest_type;
        assert!(modd.is_empty());
        let numeric_cast = match (source_l, dest_l) {
            (ref ty1, ref ty2) if ty1 == ty2 => None,
            (&TypeLayout::Scalar(ref s1), &TypeLayout::Scalar(ref s2)) => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Scalar));
                Some(cast)
            },
            (&TypeLayout::Vector(ref s1, ref x1), &TypeLayout::Vector(ref s2, ref x2)) if x1 == x2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Vector(*x2)));
                Some(cast)
            },
            (&TypeLayout::Matrix(ref s1, ref x1, ref y1), &TypeLayout::Matrix(ref s2, ref x2, ref y2)) if x1 == x2 && y1 == y2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Matrix(*x2, *y2)));
                Some(cast)
            },
            // Vector <-> Matrix casts not implemented
            // Struct casts only supported for same type structs
            _ => return Err(()),
        };
        Ok(ImplicitConversion(source_copy, value_type_cast, numeric_cast))
    }

    pub fn get_rank(&self) -> ConversionRank {
        match *self {
            ImplicitConversion(_, _, None) => ConversionRank::Exact,
            ImplicitConversion(_, _, Some(ref numeric)) => numeric.get_rank(),
        }
    }

     pub fn get_target_type(&self) -> ExpressionType {
        match *self {
            ImplicitConversion(ref source_type, None, None) => match source_type.clone() {
                ExpressionType(ty, _) => ExpressionType(ty, ValueType::Rvalue)
            },
            ImplicitConversion(_, Some(ref vtc), None) => vtc.get_target_type(),
            ImplicitConversion(_, _, Some(ref numeric)) => numeric.get_target_type(),
        }
    }

    pub fn apply(&self, expr: Expression) -> Expression {
        match *self {
            ImplicitConversion(_, _, None) => expr,
            ImplicitConversion(_, _, Some(ref cast)) => cast.apply(expr),
        }
    }
}
