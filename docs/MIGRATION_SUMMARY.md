# Migration Plan Summary

**Quick Reference**: Information mapping strategy - preserve information, not file structure.

**Key Principle**: Information should be captured in logical locations in the new IA. Files can be merged, split, or reorganized.

---

## Missing Information (Create New Pages)

| File | Location | Type | Priority |
|------|----------|------|----------|
| Performance Benchmarks | `docs/reference/performance.md` | REFERENCE | High |
| Paradigm Comparison | `docs/about/comparison.md` | EXPLANATION | High |

---

## Information to Integrate (Not Preserve as Separate Files)

| Source File | Information | Integration Point | Method |
|-------------|------------|-------------------|--------|
| `automodel_training_doc.md` | Detailed training info | `get-started/automodel.md` | Integrate via progressive disclosure |
| `megatron/models/dit/README.md` | DiT-specific details | `get-started/megatron.md` | Integrate via progressive disclosure |
| `megatron/recipes/wan/wan2.1.md` | WAN-specific details | `get-started/megatron-wan.md` OR `get-started/megatron.md` | New page OR tabs |

---

## Content Gaps to Fill

### Automodel Tutorial (`get-started/automodel.md`)
- [ ] Preprocessing modes (video vs frames) - **Add as dropdown**
- [ ] `meta.json` schema - **Add as dropdown**
- [ ] Multi-node SLURM setup - **Add as dropdown**
- [ ] Validation script details - **Add new section**
- [ ] Hardware requirements - **Add as dropdown**
- [ ] Pretraining vs fine-tuning comparison - **Add as dropdown**
- [ ] Advanced parallelization - **Add as dropdown**
- [ ] Checkpoint cleanup - **Add as dropdown**

### Megatron Tutorial (`get-started/megatron.md`)
- [ ] Sequence packing details - **Add as dropdown**
- [ ] Validation details - **Add as dropdown**
- [ ] Energon format details - **Add as dropdown**
- [ ] WAN content - **Create separate WAN guide**

---

## Progressive Disclosure Strategy

### Layer 1 (Always Visible)
- Overview, key concepts, main steps

### Layer 2 (Scannable)
- Core content, essential details, main workflows

### Layer 3 (Collapsible)
- Advanced topics → Use `:::{dropdown}`
- Alternative options → Use `:::: {tab-set}`
- Detailed explanations → Use `:::{dropdown}`

### Layer 4 (Separate Pages)
- Complete reference guides → Link to existing detailed docs

---

## MyST Directives to Use

```markdown
# Dropdowns (Layer 3 content)
:::{dropdown} Advanced Topic
:icon: info
[Detailed content here]
:::

# Tabs (Alternative options)
:::: {tab-set}
::: {tab-item} Option A
[Content A]
:::
::: {tab-item} Option B
[Content B]
:::
::::

# Cards (Navigation)
::::{grid} 1 2 2 2
:::{grid-item-card} Title
:link: target
:link-type: ref
Description
:::
::::
```

---

## Implementation Order

1. **Phase 1**: Create missing files (performance, comparison)
2. **Phase 2**: Enhance existing tutorials (add dropdowns/tabs)
3. **Phase 3**: Create WAN guide page
4. **Phase 4**: Update navigation (index pages, links)
5. **Phase 5**: Verify (content audit, link check)

---

## Quick Checklist

- [ ] Performance benchmarks page created (all info from `performance-summary.md`)
- [ ] Comparison page created OR integrated (all info from `mcore_automodel_comparision_wan21.md`)
- [ ] Automodel tutorial enhanced (all info from `automodel_training_doc.md` integrated)
- [ ] Megatron tutorial enhanced (all info from `dit/README.md` integrated)
- [ ] WAN information integrated (all info from `wan2.1.md` integrated)
- [ ] All navigation updated
- [ ] **Information audit**: All information verified (not files - verify content)
- [ ] All links working
- [ ] Progressive disclosure applied correctly
- [ ] Old files archived/removed after verification

---

**Full Plan**: See `MIGRATION_PLAN.md` for detailed implementation guide.

