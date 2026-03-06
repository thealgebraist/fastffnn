Require Import Reals Lra.
Open Scope R_scope.

Section RAdam.
  Variable beta2 : R.
  Hypothesis Hbeta2 : 0 < beta2 < 1.

  Variable rho_inf : R.
  Hypothesis Hrho_inf : rho_inf > 4.

  Variable t : R.
  Hypothesis Ht : t > 0.

  Definition r_t (rt rinf : R) : R :=
    ((rt - 4) * (rt - 2) * rinf) / ((rinf - 4) * (rinf - 2) * rt).

  (* Theorem: When the dynamic gradient length rho_t > 4, the variance correction factor is strictly positive, validating the tractability of the second moment estimator. *)
  Lemma radam_variance_tractable :
    forall rt rinf : R,
    rt > 4 -> rinf > 4 -> rt <= rinf ->
    r_t rt rinf > 0.
  Proof.
    intros rt rinf Hrt Hrinf Hle.
    unfold r_t.
    assert (rt - 4 > 0) by lra.
    assert (rt - 2 > 0) by lra.
    assert (rinf - 4 > 0) by lra.
    assert (rinf - 2 > 0) by lra.
    assert (Num : (rt - 4) * (rt - 2) * rinf > 0) by nra.
    assert (Den : (rinf - 4) * (rinf - 2) * rt > 0) by nra.
    apply Rdiv_lt_0_compat; assumption.
  Qed.
End RAdam.
