
use slp_lang_hir::*;

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
pub struct ConversionRank(NumericRank, VectorRank);

#[derive(PartialEq, Debug, Clone)]
pub enum NumericRank {
    Exact,
    Promotion,
    IntToBool,
    Conversion,
    HalfIsASecondClassCitizen,
}

#[derive(PartialEq, Debug, Clone)]
pub enum VectorRank {
    Exact,
    Expand,
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
struct DimensionCast(NumericDimension, NumericDimension);

#[derive(PartialEq, Debug, Clone)]
struct NumericCast(ScalarType, ScalarType);

#[derive(PartialEq, Debug, Clone)]
struct ValueTypeCast(Type, ValueType, ValueType);

#[derive(PartialEq, Debug, Clone)]
struct ModifierCast(TypeModifier);

#[derive(PartialEq, Debug, Clone)]
pub struct ImplicitConversion(ExpressionType,
                              Option<ValueTypeCast>,
                              Option<DimensionCast>,
                              Option<NumericCast>,
                              Option<ModifierCast>);

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
        self.0.order() * 2 + self.1.order()
    }
}

impl NumericRank {
    fn order(&self) -> u32 {
        match *self {
            NumericRank::Exact => 0,
            NumericRank::Promotion => 1,
            NumericRank::IntToBool => 2,
            NumericRank::Conversion => 3,
            NumericRank::HalfIsASecondClassCitizen => 4,
        }
    }
}

impl VectorRank {
    fn order(&self) -> u32 {
        match *self {
            VectorRank::Exact => 0,
            VectorRank::Expand => 1,
        }
    }
}


impl NumericCast {
    fn new(source: &ScalarType, dest: &ScalarType) -> Result<NumericCast, ()> {
        match *dest {
            _ if source == dest => Err(()),
            ScalarType::Bool |
            ScalarType::Int |
            ScalarType::UInt |
            ScalarType::Half |
            ScalarType::Float |
            ScalarType::Double => Ok(NumericCast(source.clone(), dest.clone())),
            ScalarType::UntypedInt => Err(()),
        }
    }

    fn get_rank(&self) -> NumericRank {
        match *self {
            NumericCast(ScalarType::Bool, ref dest) => {
                match *dest {
                    ScalarType::Bool => NumericRank::Exact,
                    ScalarType::Int | ScalarType::UInt | ScalarType::Float | ScalarType::Double => {
                        NumericRank::Conversion
                    }
                    ScalarType::Half => NumericRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::UntypedInt, ref dest) => {
                match *dest {
                    ScalarType::Int | ScalarType::UInt => NumericRank::Promotion,
                    ScalarType::Bool => NumericRank::IntToBool,
                    ScalarType::Float | ScalarType::Double | ScalarType::Half => {
                        NumericRank::Conversion
                    }
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::Int, ref dest) => {
                match *dest {
                    ScalarType::Int => NumericRank::Exact,
                    ScalarType::UInt => NumericRank::Promotion,
                    ScalarType::Bool => NumericRank::IntToBool,
                    ScalarType::Float | ScalarType::Double => NumericRank::Conversion,
                    ScalarType::Half => NumericRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::UInt, ref dest) => {
                match *dest {
                    ScalarType::UInt => NumericRank::Exact,
                    ScalarType::Int => NumericRank::Promotion,
                    ScalarType::Bool => NumericRank::IntToBool,
                    ScalarType::Float | ScalarType::Double => NumericRank::Conversion,
                    ScalarType::Half => NumericRank::HalfIsASecondClassCitizen,
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::Half, ref dest) => {
                match *dest {
                    ScalarType::Half => NumericRank::Exact,
                    ScalarType::Float | ScalarType::Double => NumericRank::Promotion,
                    ScalarType::Bool | ScalarType::Int | ScalarType::UInt => {
                        NumericRank::Conversion
                    }
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::Float, ref dest) => {
                match *dest {
                    ScalarType::Float => NumericRank::Exact,
                    ScalarType::Double => NumericRank::Promotion,
                    ScalarType::Bool | ScalarType::Int | ScalarType::UInt | ScalarType::Half => {
                        NumericRank::Conversion
                    }
                    ScalarType::UntypedInt => panic!(),
                }
            }
            NumericCast(ScalarType::Double, ref dest) => {
                match *dest {
                    ScalarType::Double => NumericRank::Exact,
                    ScalarType::Bool |
                    ScalarType::Int |
                    ScalarType::UInt |
                    ScalarType::Half |
                    ScalarType::Float => NumericRank::Conversion,
                    ScalarType::UntypedInt => panic!(),
                }
            }
        }
    }

    fn get_target_type(&self, dim: NumericDimension) -> ExpressionType {
        Type::from_layout(match dim {
            NumericDimension::Scalar => TypeLayout::Scalar(self.1.clone()),
            NumericDimension::Vector(ref x) => TypeLayout::Vector(self.1.clone(), *x),
            NumericDimension::Matrix(ref x, ref y) => TypeLayout::Matrix(self.1.clone(), *x, *y),
        })
            .to_rvalue()
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
            (&ValueType::Rvalue, &ValueType::Rvalue) | (&ValueType::Lvalue, &ValueType::Lvalue) => {
                (&source.0, &dest.0, None)
            }
            (&ValueType::Lvalue, &ValueType::Rvalue) => {
                (&source.0,
                 &dest.0,
                 Some(ValueTypeCast(source.0.clone(), ValueType::Lvalue, ValueType::Rvalue)))
            }
        };

        let &Type(ref source_l, ref mods) = source_type;
        let &Type(ref dest_l, ref modd) = dest_type;

        let dimension_cast = match (source_l, dest_l, dest.1 == ValueType::Lvalue) {
            (ref ty1, ref ty2, _) if ty1 == ty2 => None,
            // Scalar to scalar
            (&TypeLayout::Scalar(_), &TypeLayout::Scalar(_), false) => None,
            // Scalar to vector (mirror)
            (&TypeLayout::Scalar(_),
             &TypeLayout::Vector(_, ref x2),
             false) => Some(DimensionCast(NumericDimension::Scalar, NumericDimension::Vector(*x2))),
            // Single vector to vector (mirror)
            (&TypeLayout::Vector(_, 1),
             &TypeLayout::Vector(_, ref x2),
             false) => {
                Some(DimensionCast(NumericDimension::Vector(1), NumericDimension::Vector(*x2)))
            }
            // Vector first element to scalar
            (&TypeLayout::Vector(_, ref x1),
             &TypeLayout::Scalar(_),
             false) => Some(DimensionCast(NumericDimension::Vector(*x1), NumericDimension::Scalar)),
            // Vector same dimension
            (&TypeLayout::Vector(_, ref x1),
             &TypeLayout::Vector(_, ref x2),
             false)
                if x1 == x2 => None,
            // Vector cull additional elements
            (&TypeLayout::Vector(_, ref x1),
             &TypeLayout::Vector(_, ref x2),
             false)
                if x2 < x1 => {
                Some(DimensionCast(NumericDimension::Vector(*x1), NumericDimension::Vector(*x2)))
            }
            // Matrix same dimension
            (&TypeLayout::Matrix(_, ref x1, ref y1),
             &TypeLayout::Matrix(_, ref x2, ref y2),
             false)
                if x1 == x2 && y1 == y2 => None,
            // Vector <-> Matrix casts not implemented
            // Struct casts only supported for same type structs
            _ => return Err(()),
        };

        let numeric_cast = match (source_l, dest_l, dest.1 == ValueType::Lvalue) {
            (ref ty1, ref ty2, _) if ty1 == ty2 => None,
            _ => {
                let source_scalar = match *source_l {
                    TypeLayout::Scalar(ref s) => s,
                    TypeLayout::Vector(ref s, _) => s,
                    TypeLayout::Matrix(ref s, _, _) => s,
                    _ => return Err(()),
                };
                let dest_scalar = match *dest_l {
                    TypeLayout::Scalar(ref s) => s,
                    TypeLayout::Vector(ref s, _) => s,
                    TypeLayout::Matrix(ref s, _, _) => s,
                    _ => return Err(()),
                };
                if source_scalar == dest_scalar {
                    None
                } else {
                    let cast = try!(NumericCast::new(source_scalar, dest_scalar));
                    Some(cast)
                }
            }
        };

        let modifier_cast = if mods != modd {
            // Can't remove important modifiers from lvalues
            // If they're rvalues we're implicitly creating a new
            // unmodified rvalues from the source lvalue/rvalue
            // This should let us use consts + volatiles as normal
            // typed inputs, but not as out params
            if dest.1 == ValueType::Lvalue {
                if mods.is_const && !modd.is_const {
                    return Err(());
                };
                if mods.volatile && !modd.volatile {
                    return Err(());
                };
            };
            Some(ModifierCast(modd.clone()))
        } else {
            None
        };

        Ok(ImplicitConversion(source_copy,
                              value_type_cast,
                              dimension_cast,
                              numeric_cast,
                              modifier_cast))
    }

    pub fn get_rank(&self) -> ConversionRank {
        let &ImplicitConversion(_, _, ref dim_cast, ref num_cast, _) = self;
        let vec = match *dim_cast {
            None |
            Some(DimensionCast(NumericDimension::Scalar, NumericDimension::Vector(1))) |
            Some(DimensionCast(NumericDimension::Vector(1), NumericDimension::Scalar)) => {
                VectorRank::Exact
            }
            _ => VectorRank::Expand,
        };
        let num = match *num_cast {
            Some(ref numeric) => numeric.get_rank(),
            None => NumericRank::Exact,
        };
        ConversionRank(num, vec)
    }

    pub fn get_target_type(&self) -> ExpressionType {
        let &ImplicitConversion(ref source_type,
                                ref value_type_cast,
                                ref dimension_cast,
                                ref numeric_cast,
                                ref mod_cast) = self;
        let ty = source_type.clone();
        let ty = match *value_type_cast {
            Some(ref vtc) => vtc.get_target_type(),
            None => ty,
        };
        let dim = match *dimension_cast {
            Some(DimensionCast(_, ref dim)) => Some(dim.clone()),
            None => {
                match (ty.0).0 {
                    TypeLayout::Scalar(_) => Some(NumericDimension::Scalar),
                    TypeLayout::Vector(_, ref x) => Some(NumericDimension::Vector(*x)),
                    TypeLayout::Matrix(_, ref x, ref y) => Some(NumericDimension::Matrix(*x, *y)),
                    _ => None,
                }
            }
        };
        let ty = match *numeric_cast {
            Some(ref nc) => {
                nc.get_target_type(dim.expect("expecting numeric cast to operate on numeric type"))
            }
            None => {
                match dim {
                    Some(dim) => {
                        let ExpressionType(Type(tyl, m), vt) = ty;
                        let scalar = match tyl {
                            TypeLayout::Scalar(scalar) => scalar,
                            TypeLayout::Vector(scalar, _) => scalar,
                            TypeLayout::Matrix(scalar, _, _) => scalar,
                            _ => panic!("dimension cast on non numeric type"),
                        };
                        let tyl_dim = match dim {
                            NumericDimension::Scalar => TypeLayout::Scalar(scalar),
                            NumericDimension::Vector(x) => TypeLayout::Vector(scalar, x),
                            NumericDimension::Matrix(x, y) => TypeLayout::Matrix(scalar, x, y),
                        };
                        ExpressionType(Type(tyl_dim, m), vt)
                    }
                    None => ty,
                }
            }
        };
        let ty = match *mod_cast {
            Some(ref mc) => mc.modify(ty),
            None => ty,
        };
        ty
    }

    pub fn apply(&self, expr: Expression) -> Expression {
        match *self {
            ImplicitConversion(_, _, None, None, None) => expr,
            _ => Expression::Cast(self.get_target_type().0, Box::new(expr)),
        }
    }
}

#[test]
fn test_implicitconversion() {

    let basic_types = &[Type::bool(), Type::int(), Type::uint(), Type::float(), Type::floatn(4)];

    for ty in basic_types {
        assert_eq!(ImplicitConversion::find(&ty.to_rvalue(), &ty.to_rvalue()),
                   Ok(ImplicitConversion(ty.to_rvalue(), None, None, None, None)));
        assert_eq!(ImplicitConversion::find(&ty.to_lvalue(), &ty.to_rvalue()),
                   Ok(ImplicitConversion(ty.to_lvalue(),
                                         Some(ValueTypeCast(ty.clone(),
                                                            ValueType::Lvalue,
                                                            ValueType::Rvalue)),
                                         None,
                                         None,
                                         None)));
        assert_eq!(ImplicitConversion::find(&ty.to_rvalue(), &ty.to_lvalue()),
                   Err(()));
        assert_eq!(ImplicitConversion::find(&ty.to_lvalue(), &ty.to_lvalue()),
                   Ok(ImplicitConversion(ty.to_lvalue(), None, None, None, None)));
    }

    assert_eq!(ImplicitConversion::find(&Type::from_layout(TypeLayout::SamplerState).to_lvalue(),
                                        &Type::uint().to_lvalue()),
               Err(()));
    assert_eq!(ImplicitConversion::find(&Type::from_object(ObjectType::Buffer(DataType(DataLayout::Vector(ScalarType::Float, 4), TypeModifier::default()))).to_lvalue(), &Type::uint().to_lvalue()), Err(()));

    assert_eq!(ImplicitConversion::find(&Type::int().to_rvalue(), &Type::intn(1).to_rvalue()),
               Ok(ImplicitConversion(Type::int().to_rvalue(),
                                     None,
                                     Some(DimensionCast(NumericDimension::Scalar,
                                                        NumericDimension::Vector(1))),
                                     None,
                                     None)));
    assert_eq!(ImplicitConversion::find(&Type::intn(1).to_rvalue(), &Type::int().to_rvalue()),
               Ok(ImplicitConversion(Type::intn(1).to_rvalue(),
                                     None,
                                     Some(DimensionCast(NumericDimension::Vector(1),
                                                        NumericDimension::Scalar)),
                                     None,
                                     None)));
    assert_eq!(ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(4).to_rvalue()),
               Ok(ImplicitConversion(Type::uint().to_rvalue(),
                                     None,
                                     Some(DimensionCast(NumericDimension::Scalar,
                                                        NumericDimension::Vector(4))),
                                     None,
                                     None)));
    assert_eq!(ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uintn(4).to_rvalue()),
               Ok(ImplicitConversion(Type::uintn(1).to_rvalue(),
                                     None,
                                     Some(DimensionCast(NumericDimension::Vector(1),
                                                        NumericDimension::Vector(4))),
                                     None,
                                     None)));
}

#[test]
fn test_get_rank() {
    assert_eq!(ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(1).to_rvalue())
                   .unwrap()
                   .get_rank(),
               ConversionRank(NumericRank::Exact, VectorRank::Exact));
    assert_eq!(ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uint().to_rvalue())
                   .unwrap()
                   .get_rank(),
               ConversionRank(NumericRank::Exact, VectorRank::Exact));
    assert_eq!(ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(4).to_rvalue())
                   .unwrap()
                   .get_rank(),
               ConversionRank(NumericRank::Exact, VectorRank::Expand));
    assert_eq!(ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uintn(4).to_rvalue())
                   .unwrap()
                   .get_rank(),
               ConversionRank(NumericRank::Exact, VectorRank::Expand));
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
            None,
            None
        ))
    );
    // const to non-const lvalue
    assert_eq!(ImplicitConversion::find(&(Type(TypeLayout::from_scalar(ScalarType::Int),
                                               TypeModifier::const_only())
                                              .to_lvalue()),
                                        &(Type(TypeLayout::from_scalar(ScalarType::Int),
                                               TypeModifier::default())
                                              .to_lvalue())),
               Err(()));
}
