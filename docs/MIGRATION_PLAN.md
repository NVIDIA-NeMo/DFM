# Documentation Migration Plan: Preserving All Information

**Goal**: Capture all information from old docs in the new information architecture, organized logically using Diataxis, progressive disclosure, and MyST directives.

**Status**: Draft Plan  
**Date**: 2025-01-XX

**Key Principle**: Preserve **information**, not file structure. Content can be merged, split, or reorganized as long as all information is captured in a well-organized manner.

---

## Overview

This plan ensures:
- ✅ **Zero information loss**: All content from old docs preserved somewhere logical
- ✅ **Mature information architecture**: Content organized by purpose and user need
- ✅ **Diataxis alignment**: Content organized by type (Tutorial, How-To, Explanation, Reference)
- ✅ **Progressive disclosure**: Advanced details in dropdowns/tabs/separate pages
- ✅ **Cognitive load reduction**: Scannable structure with clear navigation

---

## Information Inventory (Not File Inventory)

### Information Currently Missing from New Structure

1. **Performance Benchmarks**
   - **Source**: `performance-summary.md`
   - **Information**: Nomenclature, metrics, benchmark tables (DGX-GB200, GB300, H100)
   - **Best Location**: `docs/reference/performance.md` (REFERENCE type)
   - **Status**: Missing entirely

2. **Paradigm Comparison Analysis**
   - **Source**: `mcore_automodel_comparision_wan21.md`
   - **Information**: Experimental comparison, training curves, caveats
   - **Best Location**: `docs/about/comparison.md` OR integrate into `docs/about/concepts/training-paradigms.md`
   - **Status**: Missing entirely

### Information in Orphaned Files (Needs Integration)

1. **Detailed Automodel Training Information**
   - **Source**: `automodel_training_doc.md`
   - **Information**: Preprocessing modes, validation, hardware reqs, advanced config
   - **Best Location**: Integrate into `get-started/automodel.md` (progressive disclosure)
   - **Status**: Exists but not integrated

2. **DiT-Specific Training Details**
   - **Source**: `megatron/models/dit/README.md`
   - **Information**: Sequence packing details, Energon format, validation
   - **Best Location**: Integrate into `get-started/megatron.md` (progressive disclosure)
   - **Status**: Exists but not integrated

3. **WAN-Specific Training Information**
   - **Source**: `megatron/recipes/wan/wan2.1.md`
   - **Information**: WAN dataset prep, training modes, WAN-specific workflows
   - **Best Location**: Either:
     - Option A: `get-started/megatron-wan.md` (separate guide)
     - Option B: Enhance `get-started/megatron.md` with WAN section (tabs)
   - **Status**: Exists but not integrated

---

## Information Mapping Strategy

**Approach**: Map information to logical locations in new IA, not files to files.

### Information Organization Principles

1. **User Intent First**: Where would users look for this information?
2. **Diataxis Alignment**: What type of content is this? (Tutorial/How-To/Explanation/Reference)
3. **Progressive Disclosure**: What layer does this belong to? (Core/Advanced/Reference)
4. **Logical Grouping**: Related information should be together

## Migration Strategy by Information Type

### 1. Performance Summary (`performance-summary.md` → `docs/reference/performance.md`)

**Diataxis Type**: REFERENCE  
**Progressive Disclosure**: Use tabs for different systems, dropdowns for detailed metrics

**Structure**:
```markdown
# Performance Benchmarks

## Overview
[Layer 1: 30-second overview]

## Nomenclature
[Layer 2: Core definitions - use dropdowns for detailed explanations]

## Performance Metrics
[Layer 2: Core metrics explanation]

## Benchmark Results
[Layer 2: Main results - use tabs for different systems]

:::: {tab-set}
::: {tab-item} DGX-GB200
[Results table]
:::
::: {tab-item} DGX-GB300
[Results table]
:::
::: {tab-item} DGX-H100
[Results table]
:::
::::

## Detailed Configurations
[Layer 3: Advanced details in dropdowns]
```

**Content to Preserve**:
- ✅ All nomenclature definitions (GBS, MBS, FSDP, TP, SP, PP, CP, VP, EP)
- ✅ Performance metrics explanation (Tokens/sec/GPU, Model TFLOP/sec/GPU)
- ✅ All benchmark tables (DGX-GB200, DGX-GB300, DGX-H100)
- ✅ Both Megatron-Core and NeMo Automodel results
- ✅ All model configurations

**Progressive Disclosure**:
- **Layer 1**: Overview + summary table
- **Layer 2**: Core metrics + main results (tabs for systems)
- **Layer 3**: Detailed configurations (dropdowns)
- **Layer 4**: Raw data tables (if needed, separate page)

---

### 2. Comparison Document (`mcore_automodel_comparision_wan21.md` → `docs/about/comparison.md`)

**Diataxis Type**: EXPLANATION  
**Progressive Disclosure**: Use tabs for stages, dropdowns for detailed analysis

**Structure**:
```markdown
# Automodel vs Megatron Comparison

## Overview
[Layer 1: What this comparison shows]

## Experiment Overview
[Layer 2: Core experiment details]

## Training Stages
[Layer 2: Use tabs for Stage 1 vs Stage 2]

:::: {tab-set}
::: {tab-item} Stage 1: Text-to-Image
[Dataset, setup, results]
:::
::: {tab-item} Stage 2: Text-to-Video
[Dataset, setup, results]
:::
::::

## Results Analysis
[Layer 2: Training curves with images]

:::{dropdown} Detailed Analysis
[Layer 3: Caveats and technical details]
:::

## Key Takeaways
[Layer 2: Summary comparison]
```

**Content to Preserve**:
- ✅ Complete experiment overview
- ✅ Both training stages (Text→Image, Text→Video)
- ✅ Dataset details (3K videos, 120K images)
- ✅ Training setup comparison tables
- ✅ Training curve images (both stages)
- ✅ Important caveat about Megatron-Core timestep handling
- ✅ All parallelism configurations

**Progressive Disclosure**:
- **Layer 1**: Overview + key findings
- **Layer 2**: Main comparison (tabs for stages)
- **Layer 3**: Detailed analysis (dropdowns)
- **Layer 4**: Full technical details (if needed)

**Integration**: Also enhance `docs/about/concepts/training-paradigms.md` with link to this comparison.

---

### 3. Automodel Training Doc (`automodel_training_doc.md` → Enhance `get-started/automodel.md`)

**Diataxis Type**: TUTORIAL (enhanced)  
**Progressive Disclosure**: Add missing details as dropdowns and expandable sections

**Missing Content to Add**:

#### A. Preprocessing Details (Add to Step 1)
```markdown
### 1. Prepare Your Dataset

[Current content...]

:::{dropdown} Detailed Preprocessing Modes
[Layer 3: Full explanation of video vs frames mode]

**Full Video Mode** (`--mode video`):
- What it is: [detailed explanation]
- When to use: [use cases]
- Output: [what gets created]

**Extract Frames Mode** (`--mode frames`):
- What it is: [detailed explanation]
- When to use: [use cases]
- Output: [what gets created]
:::

:::{dropdown} meta.json Format Specification
[Layer 3: Complete schema]

```json
[Full JSON schema with all fields]
```
:::
```

#### B. Multi-Node Setup (Add to Step 3)
```markdown
### 3. Run Training

[Current single-node content...]

:::{dropdown} Multi-Node with SLURM
[Layer 3: Advanced setup]

[Complete SLURM script from old docs]
:::
```

#### C. Validation (Add new section)
```markdown
### 4. Validate Training

[New section with validation script details]

:::{dropdown} Validation Script Details
[Layer 3: Advanced validation options]

[Complete validation documentation]
:::
```

#### D. Hardware Requirements (Add as dropdown)
```markdown
:::{dropdown} Hardware Requirements
[Layer 3: System requirements]

| Component | Minimum | Recommended |
|-----------|---------|-------------|
[Full table from old docs]
:::
```

#### E. Advanced Configuration (Add as new section)
```markdown
## Advanced Topics

:::{dropdown} Pretraining vs Fine-tuning
[Layer 3: Comparison table]

[Full comparison table]
:::

:::{dropdown} Custom Parallelization
[Layer 3: Advanced parallelism]

[Custom parallelization examples]
:::

:::{dropdown} Checkpoint Management
[Layer 3: Advanced checkpointing]

[Checkpoint cleanup code]
:::
```

**Content to Preserve**:
- ✅ All preprocessing mode details
- ✅ Complete `meta.json` schema
- ✅ Multi-node SLURM setup
- ✅ Validation script documentation
- ✅ Hardware requirements table
- ✅ Pretraining vs fine-tuning comparison
- ✅ Advanced parallelization examples
- ✅ Checkpoint cleanup utilities
- ✅ Supported models table

**Progressive Disclosure**:
- **Layer 1**: Core tutorial steps (current)
- **Layer 2**: Essential details (expand current sections)
- **Layer 3**: Advanced topics (dropdowns)
- **Layer 4**: Complete reference (link to detailed guide)

**Integration Strategy**:
- Keep current tutorial structure (Layer 1-2)
- Add missing information as progressive disclosure elements (Layer 3)
- **No need to preserve `automodel_training_doc.md` as separate file** - all information integrated

---

### 4. DiT Model Guide (`megatron/models/dit/README.md` → Enhance `get-started/megatron.md`)

**Diataxis Type**: TUTORIAL (enhanced)  
**Progressive Disclosure**: Add DiT-specific details as expandable sections

**Missing Content to Add**:

#### A. Sequence Packing Details (Enhance existing section)
```markdown
### Sequence Packing

[Current brief mention...]

:::{dropdown} Understanding Sequence Packing
[Layer 3: Detailed explanation]

[Complete sequence packing explanation from old docs]
- Why use it
- How it works
- Configuration requirements
- Performance impact
:::

:::{dropdown} Sequence Packing Parameters
[Layer 3: Advanced configuration]

**Key Parameters**:
- `task_encoder_seq_length`: [explanation]
- `packing_buffer_size`: [explanation]
- `qkv_format=thd`: [why required]
:::
```

#### B. Validation Details (Add new section)
```markdown
### Monitor Training

[Current content...]

:::{dropdown} Validation and Sample Generation
[Layer 3: Advanced monitoring]

[Complete validation details from old docs]
- How validation works
- Sample generation
- WandB integration
- VAE cache requirements
:::
```

#### C. Energon Dataset Details (Enhance existing section)
```markdown
### Prepare Dataset

[Current butterfly example...]

:::{dropdown} Understanding Energon Format
[Layer 3: Advanced data format]

[Complete Energon explanation]
- WebDataset format
- Sample structure
- Energon prepare command details
:::
```

**Content to Preserve**:
- ✅ Complete sequence packing explanation
- ✅ Sequence packing parameters (`task_encoder_seq_length`, `packing_buffer_size`)
- ✅ Validation details (sample generation, WandB)
- ✅ VAE cache folder requirements
- ✅ Energon dataset format details
- ✅ Complete Energon prepare workflow
- ✅ All configuration examples

**Progressive Disclosure**:
- **Layer 1**: Core tutorial (current)
- **Layer 2**: Essential DiT details (expand current)
- **Layer 3**: Advanced topics (dropdowns)
- **Layer 4**: Complete reference (link to `dit/README.md`)

**Integration Strategy**:
- Enhance existing Megatron tutorial with DiT-specific details
- Use dropdowns for advanced topics
- **No need to preserve `dit/README.md` as separate file** - all information integrated

---

### 5. WAN Recipe Guide (`megatron/recipes/wan/wan2.1.md` → New page or enhance tutorial)

**Diataxis Type**: HOW-TO  
**Progressive Disclosure**: Use tabs for different workflows, dropdowns for details

**Decision**: Create separate WAN guide page OR enhance Megatron tutorial with WAN section

**Option A: Separate WAN Guide Page** (Recommended)
```
docs/get-started/megatron-wan.md
```

**Option B: Enhance Megatron Tutorial** (Alternative)
Add WAN section with tabs: `:::: {tab-set}` for DiT vs WAN

**Recommended Structure** (Option A):
```markdown
# Megatron WAN Workflow

## Overview
[Layer 1: What WAN is, when to use it]

## Choose Your Model
[Layer 2: DiT vs WAN decision]

:::: {tab-set}
::: {tab-item} DiT Model
:link: megatron
[Link to DiT tutorial]
:::
::: {tab-item} WAN Model
[WAN-specific content]
:::
::::

## Prepare WAN Dataset
[Layer 2: WAN-specific dataset prep]

:::{dropdown} Understanding WAN Data Format
[Layer 3: Detailed format explanation]
:::

## Train WAN Model
[Layer 2: WAN training]

:::{dropdown} Training Mode Presets
[Layer 3: pretrain vs finetune modes]

[Complete explanation of presets]
:::

:::{dropdown} Sequence Packing for WAN
[Layer 3: WAN-specific packing]

[WAN sequence packing details]
:::

## Generate Videos
[Layer 2: WAN inference]

## Parallelism Support
[Layer 2: WAN parallelism table]
```

**Content to Preserve**:
- ✅ Complete WAN overview
- ✅ WAN dataset preparation (Energon workflow)
- ✅ Training mode presets (pretrain vs finetune)
- ✅ Sequence packing for WAN
- ✅ WAN inference details
- ✅ Parallelism support table
- ✅ All configuration examples
- ✅ Mock dataset configuration

**Progressive Disclosure**:
- **Layer 1**: Overview + quick start
- **Layer 2**: Core workflow steps
- **Layer 3**: Advanced topics (dropdowns)
- **Layer 4**: Complete reference (link to `wan2.1.md`)

**Integration Strategy**:
- **Decision**: Choose Option A (separate page) OR Option B (tabs in existing tutorial)
- If Option A: Create `docs/get-started/megatron-wan.md` and integrate all WAN information
- If Option B: Add WAN section to `docs/get-started/megatron.md` using tabs
- **No need to preserve `wan2.1.md` as separate file** - all information integrated into chosen location

---

## Navigation Updates

### Update `docs/get-started/index.md`

Add WAN option:
```markdown
:::: {grid} 1 2 2 2
:::{grid-item-card} 2a. Automodel Tutorial
[Current content]
:::
:::{grid-item-card} 2b. Megatron DiT Tutorial
[Current content]
:::
:::{grid-item-card} 2c. Megatron WAN Tutorial
:link: megatron-wan
:link-type: doc
Train WAN models with Megatron for video generation.
+++
{bdg-secondary}`wan` {bdg-secondary}`megatron`
:::
::::
```

### Update `docs/about/concepts/training-paradigms.md`

Add comparison link:
```markdown
## Learn More

- [Automodel vs Megatron Comparison](comparison.md) - Detailed experimental comparison
- [Performance Benchmarks](../reference/performance.md) - Training performance metrics
```

### Update `docs/reference/index.md`

Add performance link:
```markdown
## Performance and Benchmarks

:::{grid-item-card} Performance Benchmarks
:link: performance
:link-type: doc
Training throughput and performance metrics across GPU systems.
+++
{bdg-secondary}`benchmarks` {bdg-secondary}`performance`
:::
```

---

## Implementation Checklist

### Phase 1: Create Missing Files

- [ ] **Create `docs/reference/performance.md`**
  - [ ] Migrate nomenclature section
  - [ ] Migrate performance metrics explanation
  - [ ] Migrate all benchmark tables (use tabs for systems)
  - [ ] Add progressive disclosure (dropdowns for details)
  - [ ] Add frontmatter with proper metadata
  - [ ] Link from reference index

- [ ] **Create `docs/about/comparison.md`**
  - [ ] Migrate experiment overview
  - [ ] Migrate training stages (use tabs)
  - [ ] Migrate training curves (include images)
  - [ ] Migrate caveats and analysis
  - [ ] Add progressive disclosure
  - [ ] Add frontmatter with proper metadata
  - [ ] Link from training-paradigms page

### Phase 2: Integrate Information into Existing Tutorials

- [ ] **Enhance `docs/get-started/automodel.md`**
  - [ ] Integrate preprocessing details (dropdown)
  - [ ] Integrate `meta.json` schema (dropdown)
  - [ ] Integrate multi-node SLURM setup (dropdown)
  - [ ] Integrate validation section
  - [ ] Integrate hardware requirements (dropdown)
  - [ ] Integrate advanced topics section (dropdowns)
  - [ ] **Archive or remove `automodel_training_doc.md`** (information now integrated)

- [ ] **Enhance `docs/get-started/megatron.md`**
  - [ ] Integrate sequence packing details (dropdown)
  - [ ] Integrate validation details (dropdown)
  - [ ] Integrate Energon format details (dropdown)
  - [ ] **Archive or remove `megatron/models/dit/README.md`** (information now integrated)

### Phase 3: Integrate WAN Information

- [ ] **Decide**: Separate WAN guide OR tabs in Megatron tutorial
- [ ] **If separate guide**: Create `docs/get-started/megatron-wan.md`
  - [ ] Integrate all WAN information
  - [ ] Add progressive disclosure
  - [ ] **Archive or remove `megatron/recipes/wan/wan2.1.md`** (information now integrated)
- [ ] **If tabs**: Enhance `docs/get-started/megatron.md`
  - [ ] Add WAN section with tabs (DiT vs WAN)
  - [ ] Integrate all WAN information
  - [ ] **Archive or remove `megatron/recipes/wan/wan2.1.md`** (information now integrated)

### Phase 4: Update Navigation

- [ ] **Update `docs/get-started/index.md`**
  - [ ] Add WAN tutorial card
  - [ ] Update comparison table

- [ ] **Update `docs/about/concepts/training-paradigms.md`**
  - [ ] Add comparison link
  - [ ] Add performance link

- [ ] **Update `docs/reference/index.md`**
  - [ ] Add performance benchmarks card

- [ ] **Update `docs/index.md`** (if needed)
  - [ ] Ensure all new pages are discoverable

### Phase 5: Verify Content Preservation

- [ ] **Content Audit**
  - [ ] Verify all nomenclature preserved
  - [ ] Verify all tables preserved
  - [ ] Verify all code examples preserved
  - [ ] Verify all images preserved
  - [ ] Verify all configuration examples preserved
  - [ ] Verify all troubleshooting content preserved

- [ ] **Link Verification**
  - [ ] All internal links work
  - [ ] All reference targets exist
  - [ ] All images load correctly
  - [ ] All code examples render

- [ ] **Progressive Disclosure Check**
  - [ ] Layer 1 content scannable in 30 seconds
  - [ ] Layer 2 content accessible without scrolling
  - [ ] Layer 3 content in dropdowns/tabs
  - [ ] Layer 4 content linked appropriately

---

## Progressive Disclosure Patterns

### Pattern 1: Advanced Details → Dropdown
```markdown
## Core Concept

[Layer 2: Essential explanation]

:::{dropdown} Advanced: Detailed Analysis
[Layer 3: Full technical details]
:::
```

### Pattern 2: Alternative Options → Tabs
```markdown
## Choose Your Approach

:::: {tab-set}
::: {tab-item} Option A
[Content for option A]
:::
::: {tab-item} Option B
[Content for option B]
:::
::::
```

### Pattern 3: Reference Material → Separate Page + Link
```markdown
## Core Tutorial

[Layer 1-2: Essential steps]

## Complete Reference

For complete configuration options and advanced topics, see:
[Complete Reference Guide](reference-guide.md)
```

### Pattern 4: Comparison Tables → Collapsible
```markdown
## Quick Comparison

[Layer 2: Summary table]

:::{dropdown} Detailed Comparison
[Layer 3: Full comparison with all details]
:::
```

---

## Information Mapping to New IA

| Information Source | Information Type | New Location | Diataxis Type | Integration Method |
|-------------------|------------------|--------------|---------------|-------------------|
| `performance-summary.md` | Performance benchmarks | `docs/reference/performance.md` | REFERENCE | New page (all info) |
| `mcore_automodel_comparision_wan21.md` | Paradigm comparison | `docs/about/comparison.md` OR `docs/about/concepts/training-paradigms.md` | EXPLANATION | New page OR integrate |
| `automodel_training_doc.md` | Detailed training info | `docs/get-started/automodel.md` | TUTORIAL | Integrate (progressive disclosure) |
| `megatron/models/dit/README.md` | DiT-specific details | `docs/get-started/megatron.md` | TUTORIAL | Integrate (progressive disclosure) |
| `megatron/recipes/wan/wan2.1.md` | WAN-specific details | `docs/get-started/megatron-wan.md` OR `docs/get-started/megatron.md` | TUTORIAL/HOW-TO | New page OR integrate with tabs |

---

## Content Fidelity Principles

1. **Preserve All Technical Details**
   - All configuration examples
   - All code snippets
   - All parameter explanations
   - All troubleshooting content

2. **Preserve All Data**
   - All benchmark numbers
   - All comparison tables
   - All training configurations
   - All hardware specifications

3. **Preserve All Context**
   - Experiment methodology
   - Caveats and limitations
   - Use case guidance
   - Best practices

4. **Improve Organization**
   - Group related content
   - Use progressive disclosure
   - Add clear navigation
   - Improve scannability

---

## Success Criteria

✅ **Zero Information Loss**
- All content from old docs present in new structure
- All tables, code examples, images preserved
- All technical details maintained

✅ **Improved Usability**
- Clear navigation paths
- Progressive disclosure reduces cognitive load
- Scannable structure (30-second test passes)

✅ **Diataxis Compliance**
- Each page has single clear purpose
- Content type matches user intent
- Cross-links to related types

✅ **Maintainability**
- Clear file organization
- Consistent structure
- Easy to update
- Single source of truth (new IA)

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize phases** (suggest: Phase 1 → 2 → 3 → 4 → 5)
3. **Execute migration** following checklist
4. **Verify information** using audit checklist (verify all info captured, not files)
5. **Test navigation** and user flows
6. **Archive old files** after verification (information is now in new IA)

---

## Notes

- **Information Preservation**: Focus on preserving information, not file structure
- **File Cleanup**: After integration, old files can be archived or removed (information is captured)
- **Images**: Ensure all images copied to new locations with correct paths
- **Links**: Update all internal links to new structure
- **Frontmatter**: Add consistent frontmatter to all new/modified files
- **Testing**: Build docs locally to verify all MyST directives render correctly
- **Mature IA**: The new structure should be the source of truth; old files are temporary

