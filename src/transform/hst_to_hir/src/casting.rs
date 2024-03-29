use crate::pel;
use slp_lang_hir::ScalarType::*;
use slp_lang_hir::ValueType::Lvalue;
use slp_lang_hir::ValueType::Rvalue;
use slp_lang_hir::*;

// Overload priority
// =================
//
// Testing the priority of functions when passed arguments of a given type:
//
// bool(1)              bool     -> uint/int/float/double                              -> half
// int(2)               int      -> uint                  -> bool -> float/double      -> half
// untyped int literal:             uint/int              -> bool -> float/double/half
// uint(1)              uint     -> int                   -> bool -> float/double      -> half
// half:                half     -> float/double                  -> bool/int/uint
// float:               float    -> double                        -> bool/int/uint/half
// double:              double                                    -> bool/int/uint/float/half
//
// (1): Including exact literals
// (2): Including type casted int literals
//
// lvalue / rvalue didn't seem to make a difference. These don't seem consistent
// with C or C++ rules.
//
// Priority:
// 1) Exact matches fit first (untyped int literal could be either)
// 2) Promotion
// 3) Bool if the source is an int
// 4) Any convertible type, except:
// 5) Halfs if you're an bool/int/uint, unless it's an untyped literal
//    (something to do with halfs being smaller than int/uint but not literals???)
//
// I was going to try to implement nice logic in here, but I honestly have no
// idea what rules govern these priorities.

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
    /// Same dimension
    Exact,
    /// Expand a scalar to fill all slots in a vector
    Expand,
    /// Cull the later elements in the vector
    Contract,
}

#[derive(PartialEq, Debug, Clone)]
pub enum ConversionPriority {
    Better,
    Equal,
    Worse,
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
pub struct ImplicitConversion(
    ExpressionType,
    Option<ValueTypeCast>,
    Option<DimensionCast>,
    Option<NumericCast>,
    Option<ModifierCast>,
);

impl ConversionRank {
    pub fn get_numeric_rank(&self) -> &NumericRank {
        &self.0
    }
    pub fn get_vector_rank(&self) -> &VectorRank {
        &self.1
    }
}

impl NumericRank {
    pub fn compare(&self, other: &NumericRank) -> ConversionPriority {
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
            NumericRank::Exact => 0,
            NumericRank::Promotion => 1,
            NumericRank::IntToBool => 2,
            NumericRank::Conversion => 3,
            NumericRank::HalfIsASecondClassCitizen => 4,
        }
    }
}

impl VectorRank {
    pub fn worst_to_best() -> &'static [VectorRank] {
        const PRIO: &'static [VectorRank] =
            &[VectorRank::Contract, VectorRank::Expand, VectorRank::Exact];
        PRIO
    }
}

impl NumericCast {
    fn new(source: &ScalarType, dest: &ScalarType) -> Result<NumericCast, ()> {
        match *dest {
            _ if source == dest => Err(()),
            Bool | Int | UInt | Half | Float | Double => {
                Ok(NumericCast(source.clone(), dest.clone()))
            }
            UntypedInt => Err(()),
        }
    }

    fn get_rank(&self) -> NumericRank {
        match *self {
            NumericCast(Bool, ref dest) => match *dest {
                Bool => NumericRank::Exact,
                Int | UInt | Float | Double => NumericRank::Conversion,
                Half => NumericRank::HalfIsASecondClassCitizen,
                UntypedInt => panic!(),
            },
            NumericCast(UntypedInt, ref dest) => match *dest {
                Int | UInt => NumericRank::Promotion,
                Bool => NumericRank::IntToBool,
                Float | Double | Half => NumericRank::Conversion,
                UntypedInt => panic!(),
            },
            NumericCast(Int, ref dest) => match *dest {
                Int => NumericRank::Exact,
                UInt => NumericRank::Promotion,
                Bool => NumericRank::IntToBool,
                Float | Double => NumericRank::Conversion,
                Half => NumericRank::HalfIsASecondClassCitizen,
                UntypedInt => panic!(),
            },
            NumericCast(UInt, ref dest) => match *dest {
                UInt => NumericRank::Exact,
                Int => NumericRank::Promotion,
                Bool => NumericRank::IntToBool,
                Float | Double => NumericRank::Conversion,
                Half => NumericRank::HalfIsASecondClassCitizen,
                UntypedInt => panic!(),
            },
            NumericCast(Half, ref dest) => match *dest {
                Half => NumericRank::Exact,
                Float | Double => NumericRank::Promotion,
                Bool | Int | UInt => NumericRank::Conversion,
                UntypedInt => panic!(),
            },
            NumericCast(Float, ref dest) => match *dest {
                Float => NumericRank::Exact,
                Double => NumericRank::Promotion,
                Bool | Int | UInt | Half => NumericRank::Conversion,
                UntypedInt => panic!(),
            },
            NumericCast(Double, ref dest) => match *dest {
                Double => NumericRank::Exact,
                Bool | Int | UInt | Half | Float => NumericRank::Conversion,
                UntypedInt => panic!(),
            },
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
            (&Rvalue, &Lvalue) => return Err(()),
            (&Rvalue, &Rvalue) | (&Lvalue, &Lvalue) => (&source.0, &dest.0, None),
            (&Lvalue, &Rvalue) => (
                &source.0,
                &dest.0,
                Some(ValueTypeCast(source.0.clone(), Lvalue, Rvalue)),
            ),
        };

        let &Type(ref source_l, ref mods) = source_type;
        let &Type(ref dest_l, ref modd) = dest_type;

        let dimension_cast = match (source_l, dest_l, dest.1 == Lvalue) {
            (ref ty1, ref ty2, _) if ty1 == ty2 => None,
            // Scalar to scalar
            (&TypeLayout::Scalar(_), &TypeLayout::Scalar(_), false) => None,
            // Scalar to vector1 of same type (works for lvalues)
            (&TypeLayout::Scalar(ref s1), &TypeLayout::Vector(ref s2, ref x2), _)
                if s1 == s2 && *x2 == 1 =>
            {
                Some(DimensionCast(
                    NumericDimension::Scalar,
                    NumericDimension::Vector(1),
                ))
            }
            // vector1 to scalar of same type (works for lvalues)
            (&TypeLayout::Vector(ref s1, ref x1), &TypeLayout::Scalar(ref s2), _)
                if s1 == s2 && *x1 == 1 =>
            {
                Some(DimensionCast(
                    NumericDimension::Vector(1),
                    NumericDimension::Scalar,
                ))
            }
            // Scalar to vector (mirror)
            (&TypeLayout::Scalar(_), &TypeLayout::Vector(_, ref x2), false) => Some(DimensionCast(
                NumericDimension::Scalar,
                NumericDimension::Vector(*x2),
            )),
            // Single vector to vector (mirror)
            (&TypeLayout::Vector(_, 1), &TypeLayout::Vector(_, ref x2), false) => Some(
                DimensionCast(NumericDimension::Vector(1), NumericDimension::Vector(*x2)),
            ),
            // Vector first element to scalar
            (&TypeLayout::Vector(_, ref x1), &TypeLayout::Scalar(_), false) => Some(DimensionCast(
                NumericDimension::Vector(*x1),
                NumericDimension::Scalar,
            )),
            // Vector same dimension
            (&TypeLayout::Vector(_, ref x1), &TypeLayout::Vector(_, ref x2), false) if x1 == x2 => {
                None
            }
            // Vector cull additional elements
            (&TypeLayout::Vector(_, ref x1), &TypeLayout::Vector(_, ref x2), false) if x2 < x1 => {
                Some(DimensionCast(
                    NumericDimension::Vector(*x1),
                    NumericDimension::Vector(*x2),
                ))
            }
            // Matrix same dimension
            (
                &TypeLayout::Matrix(_, ref x1, ref y1),
                &TypeLayout::Matrix(_, ref x2, ref y2),
                false,
            ) if x1 == x2 && y1 == y2 => None,
            // Vector <-> Matrix casts not implemented
            // Struct casts only supported for same type structs
            _ => return Err(()),
        };

        let numeric_cast = match (source_l, dest_l) {
            (ref ty1, ref ty2) if ty1 == ty2 => None,
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
                    let cast = NumericCast::new(source_scalar, dest_scalar)?;
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
            if dest.1 == Lvalue {
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

        Ok(ImplicitConversion(
            source_copy,
            value_type_cast,
            dimension_cast,
            numeric_cast,
            modifier_cast,
        ))
    }

    pub fn get_rank(&self) -> ConversionRank {
        use slp_lang_hir::NumericDimension::Scalar;
        use slp_lang_hir::NumericDimension::Vector;
        let &ImplicitConversion(_, _, ref dim_cast, ref num_cast, _) = self;
        let vec = match *dim_cast {
            None
            | Some(DimensionCast(Scalar, Vector(1)))
            | Some(DimensionCast(Vector(1), Scalar)) => VectorRank::Exact,
            Some(DimensionCast(Scalar, Vector(_))) | Some(DimensionCast(Vector(1), Vector(_))) => {
                VectorRank::Expand
            }
            Some(DimensionCast(Vector(_), Scalar)) => VectorRank::Contract,
            Some(DimensionCast(Vector(ref l), Vector(ref r))) if l > r => VectorRank::Contract,
            Some(DimensionCast(from, to)) => panic!("invalid vector cast {:?} {:?}", from, to),
        };
        let num = match *num_cast {
            Some(ref numeric) => numeric.get_rank(),
            None => NumericRank::Exact,
        };
        ConversionRank(num, vec)
    }

    pub fn get_target_type(&self) -> ExpressionType {
        let &ImplicitConversion(
            ref source_type,
            ref value_type_cast,
            ref dimension_cast,
            ref numeric_cast,
            ref mod_cast,
        ) = self;
        let ty = source_type.clone();
        let ty = match *value_type_cast {
            Some(ref vtc) => vtc.get_target_type(),
            None => ty,
        };
        let dim = match *dimension_cast {
            Some(DimensionCast(_, ref dim)) => Some(dim.clone()),
            None => match (ty.0).0 {
                TypeLayout::Scalar(_) => Some(NumericDimension::Scalar),
                TypeLayout::Vector(_, ref x) => Some(NumericDimension::Vector(*x)),
                TypeLayout::Matrix(_, ref x, ref y) => Some(NumericDimension::Matrix(*x, *y)),
                _ => None,
            },
        };
        let ty = match *numeric_cast {
            Some(ref nc) => {
                nc.get_target_type(dim.expect("expecting numeric cast to operate on numeric type"))
            }
            None => match dim {
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
            },
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

    pub fn apply_pel(&self, expr: pel::Expression) -> pel::Expression {
        match *self {
            ImplicitConversion(_, _, None, None, None) => expr,
            _ => pel::Expression::Cast(self.get_target_type().0, Box::new(expr)),
        }
    }
}

#[test]
fn test_implicitconversion() {
    let basic_types = &[
        Type::bool(),
        Type::int(),
        Type::uint(),
        Type::float(),
        Type::floatn(4),
    ];

    for ty in basic_types {
        assert_eq!(
            ImplicitConversion::find(&ty.to_rvalue(), &ty.to_rvalue()),
            Ok(ImplicitConversion(ty.to_rvalue(), None, None, None, None))
        );
        assert_eq!(
            ImplicitConversion::find(&ty.to_lvalue(), &ty.to_rvalue()),
            Ok(ImplicitConversion(
                ty.to_lvalue(),
                Some(ValueTypeCast(ty.clone(), Lvalue, Rvalue)),
                None,
                None,
                None
            ))
        );
        assert_eq!(
            ImplicitConversion::find(&ty.to_rvalue(), &ty.to_lvalue()),
            Err(())
        );
        assert_eq!(
            ImplicitConversion::find(&ty.to_lvalue(), &ty.to_lvalue()),
            Ok(ImplicitConversion(ty.to_lvalue(), None, None, None, None))
        );
    }

    assert_eq!(
        ImplicitConversion::find(
            &Type::from_layout(TypeLayout::SamplerState).to_lvalue(),
            &Type::uint().to_lvalue()
        ),
        Err(())
    );
    let f4dty = DataType(DataLayout::Vector(Float, 4), TypeModifier::default());
    assert_eq!(
        ImplicitConversion::find(
            &Type::from_object(ObjectType::Buffer(f4dty)).to_lvalue(),
            &Type::uint().to_lvalue()
        ),
        Err(())
    );

    assert_eq!(
        ImplicitConversion::find(&Type::int().to_rvalue(), &Type::intn(1).to_rvalue()),
        Ok(ImplicitConversion(
            Type::int().to_rvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Scalar,
                NumericDimension::Vector(1)
            )),
            None,
            None
        ))
    );
    assert_eq!(
        ImplicitConversion::find(&Type::intn(1).to_rvalue(), &Type::int().to_rvalue()),
        Ok(ImplicitConversion(
            Type::intn(1).to_rvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Vector(1),
                NumericDimension::Scalar
            )),
            None,
            None
        ))
    );
    assert_eq!(
        ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(4).to_rvalue()),
        Ok(ImplicitConversion(
            Type::uint().to_rvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Scalar,
                NumericDimension::Vector(4)
            )),
            None,
            None
        ))
    );
    assert_eq!(
        ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uintn(4).to_rvalue()),
        Ok(ImplicitConversion(
            Type::uintn(1).to_rvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Vector(1),
                NumericDimension::Vector(4)
            )),
            None,
            None
        ))
    );

    assert_eq!(
        ImplicitConversion::find(&Type::int().to_lvalue(), &Type::intn(1).to_lvalue()),
        Ok(ImplicitConversion(
            Type::int().to_lvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Scalar,
                NumericDimension::Vector(1)
            )),
            None,
            None
        ))
    );
    assert_eq!(
        ImplicitConversion::find(&Type::intn(1).to_lvalue(), &Type::int().to_lvalue()),
        Ok(ImplicitConversion(
            Type::intn(1).to_lvalue(),
            None,
            Some(DimensionCast(
                NumericDimension::Vector(1),
                NumericDimension::Scalar
            )),
            None,
            None
        ))
    );
}

#[test]
fn test_get_rank() {
    assert_eq!(
        ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(1).to_rvalue())
            .unwrap()
            .get_rank(),
        ConversionRank(NumericRank::Exact, VectorRank::Exact)
    );
    assert_eq!(
        ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uint().to_rvalue())
            .unwrap()
            .get_rank(),
        ConversionRank(NumericRank::Exact, VectorRank::Exact)
    );
    assert_eq!(
        ImplicitConversion::find(&Type::uint().to_rvalue(), &Type::uintn(4).to_rvalue())
            .unwrap()
            .get_rank(),
        ConversionRank(NumericRank::Exact, VectorRank::Expand)
    );
    assert_eq!(
        ImplicitConversion::find(&Type::uintn(1).to_rvalue(), &Type::uintn(4).to_rvalue())
            .unwrap()
            .get_rank(),
        ConversionRank(NumericRank::Exact, VectorRank::Expand)
    );
}

#[test]
fn test_const() {
    // Non-const to const rvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::default()),
                Rvalue
            ),
            None,
            None,
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::default()),
                Lvalue
            ),
            Some(ValueTypeCast(Type::int(), Lvalue, Rvalue)),
            None,
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    // Const to const rvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Rvalue
            ),
            None,
            None,
            None,
            None
        ))
    );
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Lvalue
            ),
            Some(ValueTypeCast(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Lvalue,
                Rvalue
            )),
            None,
            None,
            None
        ))
    );
    // Const removing from rvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_rvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Rvalue
            ),
            None,
            None,
            None,
            Some(ModifierCast(TypeModifier::default()))
        ))
    );
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_rvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Lvalue
            ),
            Some(ValueTypeCast(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Lvalue,
                Rvalue
            )),
            None,
            None,
            Some(ModifierCast(TypeModifier::default()))
        ))
    );

    // Non-const to const lvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::default()),
                Lvalue
            ),
            None,
            None,
            None,
            Some(ModifierCast(TypeModifier::const_only()))
        ))
    );
    // const to const lvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue())
        ),
        Ok(ImplicitConversion(
            ExpressionType(
                Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()),
                Lvalue
            ),
            None,
            None,
            None,
            None
        ))
    );
    // const to non-const lvalue
    assert_eq!(
        ImplicitConversion::find(
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::const_only()).to_lvalue()),
            &(Type(TypeLayout::from_scalar(Int), TypeModifier::default()).to_lvalue())
        ),
        Err(())
    );
}
