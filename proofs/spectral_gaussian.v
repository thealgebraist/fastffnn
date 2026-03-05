Require Import Reals.
Require Import List.
Open Scope R_scope.

(* Formal Specification: Spectral FFNN with Gaussian Mixture Activation *)

Parameter g_a g_b g_c : nat -> R. (* Parameters for 8 gaussians *)

Fixpoint gaussian_sum_list (x : R) (l : list nat) : R :=
  match l with
  | nil => 0.0
  | n :: ns => (g_a n) * exp(- (g_b n) * (x - g_c n) * (x - g_c n)) + gaussian_sum_list x ns
  end.

(* Theorem: The Gaussian Sum activation is non-negative if all a_i >= 0 *)
Theorem gaussian_sum_nonneg : forall x l,
  (forall n, In n l -> g_a n >= 0.0) ->
  gaussian_sum_list x l >= 0.0.
Proof.
  intros x l H.
  induction l as [| n ns IH].
  - simpl. apply Rle_ge. apply Rle_refl.
  - simpl.
    apply Rge_le.
    apply Rplus_le_le_0_compat.
    + apply Rmult_le_pos.
      * apply Rge_le. apply H. simpl. left. reflexivity.
      * apply Rlt_le. apply exp_pos.
    + apply Rge_le. apply IH.
      intros m Hm. apply H. simpl. right. apply Hm.
Qed.
