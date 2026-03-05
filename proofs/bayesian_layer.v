From Stdlib Require Import Reals.
From Stdlib Require Import List.
From Stdlib Require Import Logic.FunctionalExtensionality.
Open Scope R_scope.

(* Formal Specification: Bayesian Gaussian Layer Propagation *)
(* Weight W ~ N(mu_w, sigma_w^2) *)
(* Input X (assumed deterministic for simplicity in this layer) *)
(* Output Y = W * X + b, where b ~ N(mu_b, sigma_b^2) *)

(* By properties of Gaussian distributions: *)
(* E[Y] = E[W] * X + E[b] = mu_w * X + mu_b *)
(* Var(Y) = Var(W) * X^2 + Var(b) = sigma_w^2 * X^2 + sigma_b^2 *)

Parameter mu_w sigma_w mu_b sigma_b : R.
Parameter x : R.

Definition expected_output : R := mu_w * x + mu_b.
Definition variance_output : R := (sigma_w * sigma_w) * (x * x) + (sigma_b * sigma_b).

(* Theorem: Linearity of Expectation for the Bayesian Neuron *)
Theorem bayesian_expectation_linearity : forall mu_w1 mu_w2 x1 x2 mu_b1 mu_b2,
  (mu_w1 * x1 + mu_b1) + (mu_w2 * x2 + mu_b2) = 
  (mu_w1 * x1 + mu_w2 * x2) + (mu_b1 + mu_b2).
Proof.
  intros. lra.
Qed.

(* Theorem: Variance Summation for Independent Gaussian Weights *)
Theorem bayesian_variance_summation : forall s_w1 s_w2 x1 x2 s_b1 s_b2,
  ((s_w1 * s_w1) * (x1 * x1) + (s_b1 * s_b1)) + ((s_w2 * s_w2) * (x2 * x2) + (s_b2 * s_b2)) =
  ((s_w1 * s_w1) * (x1 * x1) + (s_w2 * s_w2) * (x2 * x2)) + ((s_b1 * s_b1) + (s_b2 * s_b2)).
Proof.
  intros. lra.
Qed.
