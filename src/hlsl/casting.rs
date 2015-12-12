
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
struct ModifierCast(TypeModifier);

#[derive(PartialEq, Debug, Clone)]
pub struct ImplicitConversion(ExpressionType, Option<ValueTypeCast>, Option<NumericCast>, Option<ModifierCast>);

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
}

impl ValueTypeCast {
    fn get_target_type(&self) -> ExpressionType {
        ExpressionType(self.0.clone(), self.2.clone())
    }
}

impl ModifierCast {
    fn modify(&self, ty: ExpressionType) -> ExpressionType {
        let ExpressionType(Type(ty, _), vt) = ty;
        ExpressionType(Type(ty, self.0.clone()), vt)
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
        let &Type(ref dest_l, ref modd) = dest_type;
        let numeric_cast = match (source_l, dest_l, dest.1 == ValueType::Lvalue) {
            (ref ty1, ref ty2, _) if ty1 == ty2 => None,
            (&TypeLayout::Scalar(ref s1), &TypeLayout::Scalar(ref s2), false) => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Scalar));
                Some(cast)
            },
            (&TypeLayout::Vector(ref s1, ref x1), &TypeLayout::Vector(ref s2, ref x2), false) if x1 == x2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Vector(*x2)));
                Some(cast)
            },
            (&TypeLayout::Matrix(ref s1, ref x1, ref y1), &TypeLayout::Matrix(ref s2, ref x2, ref y2), false) if x1 == x2 && y1 == y2 => {
                let cast = try!(NumericCast::new(s1, s2, NumericDimension::Matrix(*x2, *y2)));
                Some(cast)
            },
            // Vector <-> Matrix casts not implemented
            // Struct casts only supported for same type structs
            _ => return Err(()),
        };

        let modifier_cast = if mods != modd {
            // Can't remove important modifiers from lvalues
            // If they're rvalues we're implicitly creating a new
            // unmodified rvalues from the source lvalue/rvalue
            // This should let us use consts + volatiles as normal
            // typed inputs, but not as out params
            if dest.1 == ValueType::Lvalue {
                if mods.is_const && !modd.is_const {
                    return Err(())
                };
                if mods.volatile && !modd.volatile {
                    return Err(())
                };
            };
            Some(ModifierCast(modd.clone()))
        } else {
            None
        };

        Ok(ImplicitConversion(source_copy, value_type_cast, numeric_cast, modifier_cast))
    }

    pub fn get_rank(&self) -> ConversionRank {
        match *self {
            ImplicitConversion(_, _, None, _) => ConversionRank::Exact,
            ImplicitConversion(_, _, Some(ref numeric), _) => numeric.get_rank(),
        }
    }

     pub fn get_target_type(&self) -> ExpressionType {
        let &ImplicitConversion(ref source_type, ref value_type_cast, ref numeric_cast, ref mod_cast) = self;
        let ty = source_type.clone();
        let ty = match *value_type_cast {
            Some(ref vtc) => vtc.get_target_type(),
            None => ty,
        };
        let ty = match *numeric_cast {
            Some(ref nc) => nc.get_target_type(),
            None => ty,
        };
        let ty = match *mod_cast {
            Some(ref mc) => mc.modify(ty),
            None => ty,
        };
        ty
    }

    pub fn apply(&self, expr: Expression) -> Expression {
        let source_type = &self.0;
        match *self {
            ImplicitConversion(_, _, None, ref mod_cast) => match *mod_cast {
                Some(ref m) => Expression::Cast(Type((source_type.0).0.clone(), m.0.clone()), Box::new(expr)),
                None => expr,
            },
            ImplicitConversion(_, _, Some(ref cast), ref mod_cast) => {
                let &NumericCast(_, ref dest, ref dim) = cast;
                let to_type_data = match *dim {
                    NumericDimension::Scalar => TypeLayout::Scalar(dest.clone()),
                    NumericDimension::Vector(ref x) => TypeLayout::Vector(dest.clone(), *x),
                    NumericDimension::Matrix(ref x, ref y) => TypeLayout::Matrix(dest.clone(), *x, *y),
                };
                Expression::Cast(Type(to_type_data, match *mod_cast { Some(ref m) => m.0.clone(), None => (source_type.0).1.clone() }), Box::new(expr))
            }
        }
    }
}

#[test]
fn test_implicitconversion() {

    let basic_types = &[Type::bool(), Type::int(), Type::uint(), Type::float(), Type::floatn(4)];

    for ty in basic_types {
        assert_eq!(ImplicitConversion::find(&ty.to_rvalue(), &ty.to_rvalue()), Ok(ImplicitConversion(ty.to_rvalue(), None, None, None)));
        assert_eq!(ImplicitConversion::find(&ty.to_lvalue(), &ty.to_rvalue()), Ok(ImplicitConversion(ty.to_lvalue(), Some(ValueTypeCast(ty.clone(), ValueType::Lvalue, ValueType::Rvalue)), None, None)));
        assert_eq!(ImplicitConversion::find(&ty.to_rvalue(), &ty.to_lvalue()), Err(()));
        assert_eq!(ImplicitConversion::find(&ty.to_lvalue(), &ty.to_lvalue()), Ok(ImplicitConversion(ty.to_lvalue(), None, None, None)));
    }

    assert_eq!(ImplicitConversion::find(&Type::from_layout(TypeLayout::SamplerState).to_lvalue(), &Type::uint().to_lvalue()), Err(()));
    assert_eq!(ImplicitConversion::find(&Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Float, 4), TypeModifier::default()))).to_lvalue(), &Type::uint().to_lvalue()), Err(()));
}

#[test]
fn test_const() {
    // Non-const to const rvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()), ValueType::Rvalue),
            None,
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()), ValueType::Lvalue),
            Some(ValueTypeCast(Type::int(), ValueType::Lvalue, ValueType::Rvalue)),
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    // Const to const rvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Rvalue),
            None,
            None,
            None
        ))
    );
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Lvalue),
            Some(ValueTypeCast(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Lvalue, ValueType::Rvalue)),
            None,
            None
        ))
    );
    // Const removing from rvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Rvalue),
            None,
            None,
            Some(ModifierCast(TypeModifier::default()))
        ))
    );
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Lvalue),
            Some(ValueTypeCast(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Lvalue, ValueType::Rvalue)),
            None,
            Some(ModifierCast(TypeModifier::default()))
        ))
    );

    // Non-const to const lvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()), ValueType::Lvalue),
            None,
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    // const to const lvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()), ValueType::Lvalue),
            None,
            None,
            None
        ))
    );
    // const to non-const lvalue
    assert_eq!(ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(ScalarType::Int), TypeModifier::default()).to_lvalue())
        ),
        Err(())
    );
}
