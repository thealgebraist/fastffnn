From Stdlib Require Import Reals.
From Stdlib Require Import List.
From Stdlib Require Import Lra.
Open Scope R_scope.

(* Formal Specification: Dynamic Capacity Expansion *)
(* Goal: Prove that a network with H+1 neurons can represent any function 
   represented by a network with H neurons. *)

Parameter input_dim : nat.
Definition weight_vector := list R. (* Size = input_dim *)

Record Neuron := {
  w : weight_vector;
  b : R;
}.

Definition network (H : nat) := list Neuron.

(* Simplified output: sum of activations *)
Fixpoint forward (inputs : weight_vector) (net : list Neuron) : R :=
  match net with
  | nil => 0.0
  | n :: ns => 
      let dot_prod := fold_left (fun acc p => acc + (fst p * snd p)) (combine n.(w) inputs) 0.0 in
      let act := if Rlt_dec 0.0 (dot_prod + n.(b)) then (dot_prod + n.(b)) else 0.0 in
      act + forward inputs ns
  end.

(* Theorem: Expansion Monotonicity *)
(* For any network of size H, there exists a network of size H+1 that 
   produces identical outputs for all inputs. *)
Theorem expansion_preserves_function : forall (H : nat) (netH : list Neuron),
  length netH = H ->
  exists (netHplus1 : list Neuron),
    length netHplus1 = (H + 1)%nat /\
    (forall (inputs : weight_vector), length inputs = input_dim -> forward inputs netHplus1 = forward inputs netH).
Proof.
  intros H netH Hlen.
  (* Construct netHplus1 by adding a zero-weight neuron *)
  set (zero_w := repeat 0.0 input_dim).
  set (zero_neuron := {| w := zero_w; b := -1.0 |}). (* Bias -1 ensures act=0 *)
  exists (zero_neuron :: netH).
  split.
  - simpl. rewrite Hlen. reflexivity.
  - intros inputs InLen. simpl.
    assert (Hdot: fold_left (fun acc p => acc + fst p * snd p) (combine zero_w inputs) 0.0 = 0.0).
    { unfold zero_w. 
      (* Induction on input_dim/inputs list *)
      clear. generalize dependent inputs. induction input_dim; intros inputs InLen.
      + destruct inputs; inversion InLen. reflexivity.
      + destruct inputs; inversion InLen. simpl.
        rewrite IHn; [lra | assumption]. }
    rewrite Hdot. simpl.
    destruct (Rlt_dec 0.0 (0.0 + -1.0)); [lra | reflexivity].
Qed.
