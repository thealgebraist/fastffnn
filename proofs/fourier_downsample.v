From Coq Require Import Reals.
From Coq Require Import List.
From Coq Require Import Bool.
From Coq Require Import Logic.FunctionalExtensionality.
Open Scope R_scope.

(* Formal Specification: Fourier Downsampling Linearity *)

Parameter image : nat -> nat -> R.
Parameter dct2d : (nat -> nat -> R) -> nat -> nat -> R.
Parameter idct2d : (nat -> nat -> R) -> nat -> nat -> R.

(* Linearity Axioms for DCT/IDCT *)
Axiom dct_linear : forall (f g : nat -> nat -> R) (a b : R),
  (fun u v => dct2d (fun x y => a * f x y + b * g x y) u v) =
  (fun u v => a * (dct2d f u v) + b * (dct2d g u v)).

Axiom idct_linear : forall (f g : nat -> nat -> R) (a b : R),
  (fun x y => idct2d (fun u v => a * f u v + b * g u v) x y) =
  (fun x y => a * (idct2d f x y) + b * (idct2d g x y)).

(* Downsampling is defined as a linear operator (truncation + IDCT) *)
Definition fourier_op (img : nat -> nat -> R) (k x y : nat) : R :=
  idct2d (fun u v => if (andb (Nat.ltb u k) (Nat.ltb v k)) then dct2d img u v else 0.0) x y.

(* A linear combination of linear operators is linear *)
Theorem downsample_is_linear : forall k x y f g a b,
  fourier_op (fun i j => a * f i j + b * g i j) k x y =
  a * fourier_op f k x y + b * fourier_op g k x y.
Proof.
  intros. unfold fourier_op.
  rewrite dct_linear.
  assert (H: (fun u v => if andb (Nat.ltb u k) (Nat.ltb v k) then a * dct2d f u v + b * dct2d g u v else 0.0) =
             (fun u v => a * (if andb (Nat.ltb u k) (Nat.ltb v k) then dct2d f u v else 0.0) +
                         b * (if andb (Nat.ltb u k) (Nat.ltb v k) then dct2d g u v else 0.0))).
  { apply functional_extensionality. intros u v.
    destruct (andb (Nat.ltb u k) (Nat.ltb v k)).
    - reflexivity.
    - rewrite Rmult_0_r, Rmult_0_r, Rplus_0_r. reflexivity. }
  rewrite H.
  rewrite idct_linear.
  reflexivity.
Qed.
