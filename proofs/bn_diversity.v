From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Formal Specification: Batch Normalization Centering (N=2) *)
(* Proof that the sum of centered activations is exactly zero. *)

Theorem bn_centering_2 : forall (x1 x2 : R),
  let mu := (x1 + x2) / 2.0 in
  (x1 - mu) + (x2 - mu) = 0.0.
Proof.
  intros x1 x2 mu.
  subst mu.
  lra.
Qed.

(* Proof that sum of squares is non-negative for N=2 *)
Theorem sum_sq_2_nonneg : forall (x1 x2 : R),
  x1 * x1 + x2 * x2 >= 0.0.
Proof.
  intros x1 x2.
  assert (H1: x1 * x1 >= 0.0) by nra.
  assert (H2: x2 * x2 >= 0.0) by nra.
  lra.
Qed.
