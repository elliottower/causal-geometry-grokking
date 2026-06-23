# Causal Geometry of Circuit Edges: An Intuitive Guide

## What we did

Start with a factorized GPT-2 model. Instead of each weight matrix being a single opaque blob, every matrix is decomposed into ~1000 "factors" --- shared building blocks that are reused across all layers and projections. Think of factors as a vocabulary of directions in residual-stream space. Every weight matrix in the model is a sentence written in that vocabulary.

We wanted to understand what information flows along each edge of the IOI (Indirect Object Identification) circuit. The standard circuit picture gives you a wiring diagram --- head A talks to head B --- but says nothing about *what* A is telling B. So we did three things:

1. **DAS (Distributed Alignment Search)**: For each factor, we asked: does intervening on this factor's activation change the model's IOI behavior? This gives us a causal importance score per factor.

2. **EAP (Edge Attribution Patching)**: We traced those causal factors through the circuit graph, decomposing each edge into contributions from individual factors. Each edge becomes a 1000-dimensional vector of factor contributions.

3. **Causal inference on the factor structure**: We clustered, factored, and ran structural equation models on the edge-level factor scores to understand the geometry of information flow.

The result is a "causal geometry atlas" --- a map of what each circuit edge carries, in terms of factor subspaces.


## Three streams in every wire

Here is the finding that surprised us most. We expected circuit edges to be *typed*: duplicate-token edges carry duplicate-token signal, name-mover edges carry name-mover signal. That is not what we found.

Every circuit edge carries **three concurrent signals**, mixed together:

- **Stream 1** (34% of variance): factors that track *which tokens are duplicated*. Post-hoc, these align with what the IOI literature calls duplicate-token detection.
- **Stream 2** (22%): factors encoding *positional structure* --- where IO and S appear in the sentence.
- **Stream 3** (22%): factors implementing *S-inhibition* --- suppressing the repeated name so the model can output the non-repeated one.

These labels were not assumed. We discovered three clusters from the data (via NMF on the edge-factor matrix), then checked what each cluster's top factors encode. The alignment with known IOI roles was the *reveal*, not the starting point.

The remaining ~22% of variance is distributed across smaller modes that don't map cleanly onto any single circuit role.

This means circuit edges are not like colored wires in a harness. They are more like fiber-optic cables carrying multiplexed signals. The "colors" are only separable when you decompose into the right basis.


## The anti-signal: narrow channels, not wide highways

Standard intuition says important edges should carry *more* signal. Bigger activations, stronger gradients. We found the opposite.

**Circuit heads have lower total coupling than non-circuit heads.** The heads Wang et al. identified as doing the real work of IOI have *weaker* aggregate factor flow than random heads in the same layers.

The analogy: a highway carries more traffic than a secret tunnel, but the tunnel is the one that matters for the spy mission. Circuits don't work by maximizing throughput. They work by *restricting* flow to a narrow, task-relevant subspace. The selectivity *is* the computation. Non-circuit heads are noisy and diffuse; circuit heads are quiet and focused.

This has implications for circuit discovery. If you search for edges by total activation magnitude, you will find highways, not tunnels. The circuit signal is in the *structure* of what flows, not the *amount*.


## Alphabet vs words: factor roles are not edge modes

Weight-space analysis tells you what each factor *means* --- factor F798 encodes token identity, factor F412 responds to positional patterns, and so on. These are the factor roles. Think of them as the alphabet.

Edge-level analysis discovers something different: *usage modes*, which are combinations of factors that co-occur on edges. These are the words. A single edge mode might blend five token-identity factors with three positional factors in a specific ratio. The mode is a composite that no single factor captures alone.

We measured the overlap: ARI (Adjusted Rand Index) between factor-role clusters and edge-mode clusters is 0.028. That is barely above chance. The two decompositions are genuinely different views of the same system. Knowing the alphabet does not tell you the words. Knowing the words does not tell you which letters they use.

This matters because it means factor-level interpretability and edge-level interpretability are *complementary*, not redundant. You need both.


## Hub versus chain: two true pictures at different scales

At the edge level, the IOI circuit looks like a sequential chain. Information flows:

> duplicate-token detection --> S-inhibition --> name mover output

Stage by stage, each link in the chain transforms the signal. This is the classic IOI circuit story.

At the factor level, it looks completely different. Factor F798 (token identity) acts as a **hub**: it broadcasts to every major circuit head in parallel, not sequentially. The chain structure emerges from how *combinations* of factors shift across edges, not from any single factor's connectivity.

Both descriptions are true. The chain is what you see when you look at composite edge modes. The hub is what you see when you track individual factors. The analogy: individual letters (factors) appear everywhere in a book, but the *words* (edge modes) follow a narrative arc. The letter "e" doesn't have a plot. The sentence does.


## Translator heads: format converters at stage boundaries

Some heads sit at the boundary between circuit stages and act as *translators*. Head L5.H9 is the clearest example: it reads edge mode C1 (dominated by duplicate-token factors) on its input edges, and writes edge mode C2 (dominated by S-inhibition factors) on its output edges.

This is not a trivial relay. The head is converting information from one format to another --- from "which token is duplicated" to "which token to suppress." The factor composition of the input and output modes is measurably different. The head performs a rotation in factor space.

We find 3-4 such translator heads in the IOI circuit, and they consistently sit at the transitions between the three canonical stages (duplicate-token, S-inhibition, name-mover). They are the connective tissue that makes the chain structure possible.


## What weight space can and cannot predict

Can you recover this edge-level structure from weights alone, without running any data through the model?

Partially. The overall DAG structure --- which heads connect to which, and what subspaces they share --- is reflected in weight space. A structural equation model (SEM) fit to the circuit DAG achieves R-squared = 0.73 on predicting factor co-occurrence patterns across edges. The DAG predicts that duplicate-token subspaces flow into S-inhibition subspaces, and that prediction holds (R-squared = 0.72 for that specific path).

**But individual edges cannot be discovered from weights.** Binary classification of "is this edge in the IOI circuit?" from weight-space features gives AUROC of 0.45-0.55 --- essentially chance. The signal is in the *joint structure* of all edges together, not in any individual edge's properties.

This is the difference between structural validation and edge discovery. You can confirm that a proposed circuit DAG is consistent with weight-space geometry (validation). You cannot find the circuit by looking at weights (discovery). The tunnel does not advertise itself; you have to run the spy mission to find it.


## Scale vs structure: the 62% and the 12%

When we decompose the edge-factor matrix with PCA, the first principal component captures 62% of the variance. It's boring: it just measures "how important is this edge?" Every factor loads positively on PC1. Important edges carry more of everything.

The interesting part is PC2, which captures 12%. It separates S-inhibition factors from duplicate-token and positional factors. This is the circuit's structural axis --- the dimension along which edges actually differ in *what* they carry, not just *how much*.

After we statistically control for edge importance (remove the scale component), the S-inhibition vs positional trade-off strengthens from r = -0.57 to r = -0.72. Scale was *masking* the true interaction structure. The real finding was hiding behind the noise of importance variation.

The takeaway: 62% of the variation across edges is uninteresting (some edges just matter more). The computational structure of the circuit lives in the remaining 38%, primarily in two axes: S-inhibition vs everything else, and positional vs S-inhibition.


## Signs matter: direction encoding

In absolute values, all DAS dimensions co-activate on edges --- when one dimension is large, the others tend to be large too. No negative correlations at all.

But when we keep the signs, a completely different picture emerges. Two dimensions from the *same* factor anti-correlate in sign (r = -0.75 for dup_token dims d3 and d26; r = -0.76 for S_inhib dims d12 and d24). Their magnitudes go up together, but their signs flip.

This means the DAS dimensions encode information as *sign contrasts*. Two dimensions that co-activate in magnitude but anti-correlate in sign together encode a single bit: which direction the information points. This is exactly what you'd expect from the DAS rotation --- it finds orthogonal directions that separate counterfactual interventions, and the sign tells you which intervention is active.


## The extended circuit: L9.H3 and the 77%

Here's something we didn't expect. When we ranked all 144 attention heads by their total DAS-EAP read mass, the #1 head is L9.H3 --- a head that is *not* part of the standard IOI circuit. It carries 42 times more DAS-EAP mass than L9.H9, the primary name mover that Wang et al. identified as the circuit's output head.

In total, 77% of attention head mass and 28% from MLPs are in non-circuit components. The standard IOI circuit accounts for only 16% of the DAS-EAP mass.

Why? Because the factorized model has a shared factor bank --- all 144 heads compute with the same factors. The DAS rotation finds directions that matter for the IOI task, but those directions propagate through *all* heads, not just the IOI-specific ones. L9.H3 happens to be a strong conduit for the shared factor subspace, even though it wasn't identified as part of the manually-discovered circuit.

This doesn't mean the IOI circuit analysis is wrong. It means the DAS-EAP decomposition measures something different: the model's general information infrastructure that overlaps with the IOI task, not just the task-specific circuit.


## Writers broadcast, readers select

Writers and readers of the same edge disagree substantially about which factors matter (cosine similarity only 0.57). The three task-relevant factors (dup_token, positional, S-inhibition) are all *reader-amplified*: readers consume proportionally more of these factors than writers produce. Token identity (token_id) goes the other way --- it's written broadly but read sparingly.

The residual stream is not a symmetric channel. The embedding writes a broad signal; downstream heads selectively extract the task-relevant components. The circuit's intelligence is in what readers *choose to consume*, not in what writers produce.


## The computation is a sign flip

This is the most concrete result from the atlas. When we track the *signed* S-inhibition mass through the IOI circuit's stages, we find:

- **Early circuit (L0-L2)**: S-inhib sign ratio = -0.44. The circuit is writing *anti*-S-inhibition.
- **Mid circuit (L3-L6)**: sign ratio = -0.72. The suppression is at its strongest.
- **Late circuit (L7-L9)**: sign ratio = **+0.73**. The sign flips. Late heads write *pro*-S-inhibition.
- **Logits**: sign ratio = -0.84. The output layer inverts the signal one last time.

The sign flip from -0.72 to +0.73 between mid and late circuit *is* the IOI computation. Early/mid heads identify which name is duplicated (writing negative S-inhibition --- "this name should be suppressed"). Late heads convert this into a positive S-inhibition signal ("suppress this specific name"). The logits layer then inverts the suppression into the model's actual output ("say the *other* name").

Non-circuit heads show no sign structure at all: they carry the factors but don't participate in the sign transformation. The computation is invisible in absolute values. You have to track the signs to see it.


## Circuit role names are misleading

Here is something that caught us off guard. We expected "S-inhibition heads" to read mostly S-inhibition factors, and "name-mover heads" to read mostly name-mover factors. That is not what happens.

S-inhibition heads (L7.H3, L7.H9, L8.H6, L8.H10) read more *dup_token* (32.7%) than S-inhibition (19.3%). Name-mover heads read only 4% name-mover factor. Only dup-token heads actually read their namesake factor most (39%).

The circuit roles from Wang et al. describe what each head *does functionally* --- they identify duplicate tokens, inhibit names, move names to the output. But the factorized decomposition describes what *subspace information* flows through each head. These are different things. A head can functionally perform S-inhibition while reading primarily from the dup-token subspace, because the computation uses multiple factor signals together, not one factor in isolation.

Factor roles are the alphabet. Circuit roles are the words. And the alphabet doesn't predict the words --- ARI = 0.028 between the two (near chance).


## Cosine lies: Grassmannian distance between factor subspaces

Everything above uses cosine similarity to compare factor profiles. Cosine works for single vectors, but it has a blind spot: it compares *mean profiles*, collapsing each factor's multi-dimensional subspace into a single summary.

When we switch to **Grassmannian distance** --- which measures the angular separation between *subspaces* via principal angles --- a different picture emerges. The dup-token and positional factors have cosine similarity 0.95 on their edge profiles (nearly identical). But their Grassmannian distance is 28% of the theoretical maximum. The three main factors occupy genuinely distinct subspaces in edge space, despite looking identical in their marginal profiles.

This is like two classrooms that teach the same subjects in the same proportions (identical cosine), but assign completely different homework in each subject (different subspaces). The subspace separation is what lets downstream heads selectively extract one signal from the multiplexed stream --- you can't select "just the dup-token component" if dup-token and positional are literally the same subspace.

For single-vector comparisons, geodesic distance (arccos of cosine) doesn't change any rankings --- it's just a rescaling that makes degrees more interpretable. Same-reader edges are 19° apart (closely aligned), same-writer edges are 43° apart (nearly at population baseline), and the writer-reader transformation is 55° (past the halfway mark to orthogonal). These numbers tell the same story as cosine, just more viscerally.

The Grassmannian finding is the new one: the three-stream model has *subspace* support, not just *profile* support. The factors are genuinely separable in multi-dimensional edge space.


## The bottom line

Circuit edges are not monolithic wires. They are multiplexed channels carrying concurrent signals in different subspaces. The structure of this multiplexing --- which subspaces appear on which edges, how they transform through translator heads, how they compose into a sequential chain from parallel hubs --- is a level of description that lives between individual factors and the circuit graph.

But most of the variation across edges (62%) is just scale --- some edges matter more. The computational structure lives in the remaining 38%, organized around a primary trade-off between S-inhibition and positional/dup-token factors.

The deepest structure is in the signs. Absolute values show a universal mixture that barely changes across layers, writer types, or circuit membership. Signed values reveal the actual computation: a sign flip in S-inhibition between mid and late circuit, invisible to any analysis that takes absolute values. The circuit's intelligence is not in *what it carries* (that's uniform) but in *which direction it points* (that's where the sign flip happens).

This structure is encoded in sign contrasts between paired DAS dimensions, read selectively by downstream heads from a broadcast residual stream, and distributed across far more model components than the standard IOI circuit. Weight space encodes the *possibility* of this structure. Activation-level causal analysis reveals which possibilities the model actually uses --- and how widely they are distributed.
