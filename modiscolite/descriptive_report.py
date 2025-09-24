import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template
from typing import List, Dict, Optional
import importlib.resources

from .report import compute_per_position_ic, _plot_weights, tomtomlite_dataframe, generate_tomtom_dataframe
from . import templates


def extract_seqlet_data(modisco_h5py: str, pattern_groups: List[str]) -> Dict:
    """Extract seqlet data for descriptive analysis."""
    with h5py.File(modisco_h5py, 'r') as modisco_results:
        patterns_data = {}

        for contribution_dir_name in pattern_groups:
            if contribution_dir_name not in modisco_results.keys():
                continue

            metacluster = modisco_results[contribution_dir_name]

            def sort_key(x):
                return int(x[0].split("_")[-1])

            for pattern_name, pattern in sorted(metacluster.items(), key=sort_key): # type: ignore
                pattern_tag = f'{contribution_dir_name}.{pattern_name}'

                # Extract basic pattern data
                ppm = np.array(pattern['sequence'][:])
                cwm = np.array(pattern["contrib_scores"][:])
                hcwm = np.array(pattern["hypothetical_contribs"][:])

                # Extract seqlet information
                seqlets_grp = pattern['seqlets']
                n_seqlets = seqlets_grp['n_seqlets'][:][0]

                # Get seqlet positions and data if available
                seqlet_starts = seqlets_grp.get('start', [])
                seqlet_ends = seqlets_grp.get('end', [])
                seqlet_example_idx = seqlets_grp.get('example_idx', [])

                # Get seqlet contribution scores and calculate importance
                seqlet_contribs = []
                seqlet_importance = []
                if 'contrib_scores' in seqlets_grp:
                    seqlet_contribs = np.array(seqlets_grp['contrib_scores'][:])
                    # Calculate total importance as sum of absolute contribution scores
                    for seqlet_contrib in seqlet_contribs:
                        importance = np.sum(np.abs(seqlet_contrib))
                        seqlet_importance.append(importance)

                # Calculate pattern statistics
                gc_content = np.mean(ppm[:, [1, 2]])  # C and G positions
                avg_importance = np.mean(np.sum(np.abs(cwm), axis=1))

                patterns_data[pattern_tag] = {
                    'ppm': ppm,
                    'cwm': cwm,
                    'hcwm': hcwm,
                    'n_seqlets': n_seqlets,
                    'gc_content': gc_content,
                    'avg_importance': avg_importance,
                    'seqlet_starts': np.array(seqlet_starts) if len(seqlet_starts) > 0 else np.array([]),
                    'seqlet_ends': np.array(seqlet_ends) if len(seqlet_ends) > 0 else np.array([]),
                    'seqlet_example_idx': np.array(seqlet_example_idx) if len(seqlet_example_idx) > 0 else np.array([]),
                    'seqlet_importance': np.array(seqlet_importance) if len(seqlet_importance) > 0 else np.array([]),
                    'seqlet_contribs': seqlet_contribs
                }

    return patterns_data


def create_comprehensive_logos(patterns_data: Dict, output_dir: str, trim_threshold: float = 0.3) -> Dict:
    """Create logo visualizations for each pattern."""
    logo_dir = os.path.join(output_dir, 'comprehensive_logos')
    os.makedirs(logo_dir, exist_ok=True)

    logo_paths = {}

    for pattern_tag, data in patterns_data.items():
        pattern_dir = os.path.join(logo_dir, pattern_tag)
        os.makedirs(pattern_dir, exist_ok=True)

        cwm = data['cwm']
        hcwm = data['hcwm']
        ppm = data['ppm']

        # Calculate trimmed version
        score = np.sum(np.abs(cwm), axis=1)
        trim_thresh = np.max(score) * trim_threshold
        pass_inds = np.where(score >= trim_thresh)[0]

        if len(pass_inds) > 0:
            start_trim = max(np.min(pass_inds) - 2, 0)
            end_trim = min(np.max(pass_inds) + 3, len(score))
            trimmed_cwm = cwm[start_trim:end_trim]
        else:
            trimmed_cwm = cwm

        # Generate logos
        logos = {}

        # CWM Logo
        cwm_path = os.path.join(pattern_dir, 'cwm_logo.png')
        _plot_weights(cwm, cwm_path, figsize=(12, 3))
        logos['cwm'] = cwm_path

        # hCWM Logo
        hcwm_path = os.path.join(pattern_dir, 'hcwm_logo.png')
        _plot_weights(hcwm, hcwm_path, figsize=(12, 3), clamp=False)
        logos['hcwm'] = hcwm_path

        # PWM Logo (using information content)
        background = np.array([0.25, 0.25, 0.25, 0.25])
        ic = compute_per_position_ic(ppm, background, 0.001)
        pwm_path = os.path.join(pattern_dir, 'pwm_logo.png')
        _plot_weights(ppm * ic[:, None], pwm_path, figsize=(12, 3))
        logos['pwm'] = pwm_path

        # Trimmed CWM Logo
        trimmed_path = os.path.join(pattern_dir, 'trimmed_cwm_logo.png')
        _plot_weights(trimmed_cwm, trimmed_path, figsize=(10, 3))
        logos['trimmed_cwm'] = trimmed_path

        logo_paths[pattern_tag] = logos

    return logo_paths


def create_distribution_plots(patterns_data: Dict, output_dir: str) -> Dict:
    """Create seqlet importance and spatial distribution plots."""
    dist_dir = os.path.join(output_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)

    distribution_paths = {}

    for pattern_tag, data in patterns_data.items():
        pattern_dir = os.path.join(dist_dir, pattern_tag)
        os.makedirs(pattern_dir, exist_ok=True)

        plots = {}

        # Seqlet importance distribution
        if len(data['seqlet_importance']) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(data['seqlet_importance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Seqlet Total Importance Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Seqlet Importance Distribution - {pattern_tag}')

            importance_path = os.path.join(pattern_dir, 'importance_distribution.png')
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['importance'] = importance_path

        # Seqlet spatial distribution
        if len(data['seqlet_starts']) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Calculate center positions
            centers = (data['seqlet_starts'] + data['seqlet_ends']) / 2
            ax.hist(centers, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Position within Input Sequence')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Seqlet Spatial Distribution - {pattern_tag}')

            spatial_path = os.path.join(pattern_dir, 'spatial_distribution.png')
            plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['spatial'] = spatial_path

        distribution_paths[pattern_tag] = plots

    return distribution_paths


def create_seqlet_example_logos(patterns_data: Dict, output_dir: str, n_examples: int = 10) -> Dict:
    """Create logo plots of seqlets from importance score quantiles to show distribution."""
    examples_dir = os.path.join(output_dir, 'seqlet_examples')
    os.makedirs(examples_dir, exist_ok=True)

    examples_paths = {}

    for pattern_tag, data in patterns_data.items():
        if len(data['seqlet_importance']) == 0 or len(data['seqlet_contribs']) == 0:
            continue

        pattern_dir = os.path.join(examples_dir, pattern_tag)
        os.makedirs(pattern_dir, exist_ok=True)

        # Get seqlets representing quantiles of importance distribution
        importance_scores = data['seqlet_importance']
        seqlet_contribs = data['seqlet_contribs']

        # Adjust n_examples if we have fewer seqlets than requested
        actual_n_examples = min(n_examples, len(importance_scores))

        # Calculate quantile percentiles (10%, 20%, ..., 100%)
        quantiles = np.linspace(10, 100, actual_n_examples)
        quantile_indices = []
        used_indices = set()

        for quantile in quantiles:
            percentile_value = np.percentile(importance_scores, quantile)
            # Find all seqlets close to this percentile value
            distances = np.abs(importance_scores - percentile_value)
            sorted_indices = np.argsort(distances)

            # Find the closest unused index
            closest_idx = None
            for idx in sorted_indices:
                if idx not in used_indices:
                    closest_idx = idx
                    break

            if closest_idx is not None:
                quantile_indices.append(closest_idx)
                used_indices.add(closest_idx)

        # Debug: print how many quantiles we found
        print(f"Pattern {pattern_tag}: Found {len(quantile_indices)} quantile examples out of {actual_n_examples} requested, from {len(importance_scores)} total seqlets")

        example_logos = []

        # Ensure we have the right number of quantile_indices
        for i in range(len(quantile_indices)):
            if i < len(quantiles):
                idx = quantile_indices[i]
                if idx < len(seqlet_contribs) and idx < len(importance_scores):
                    # Extract individual seqlet contribution scores
                    seqlet_cwm = seqlet_contribs[idx]
                    importance = importance_scores[idx]
                    quantile_pct = quantiles[i]

                    # Create logo for this seqlet (smaller height for more compact display)
                    logo_path = os.path.join(pattern_dir, f'quantile_{int(quantile_pct)}.png')

                    _plot_weights(seqlet_cwm, logo_path, figsize=(8, 1.2))

                    example_logos.append({
                        'rank': i + 1,
                        'quantile': int(quantile_pct),
                        'path': logo_path,
                        'importance': float(importance)
                    })

        examples_paths[pattern_tag] = list(reversed(example_logos))
    
    return examples_paths


def generate_descriptive_report(modisco_h5py: str, output_dir: str,
                              img_path_suffix: str = './',
                              meme_motif_db: Optional[str] = None,
                              top_n_matches: int = 3, ttl: bool = False,
                              n_examples: int = 10, trim_threshold: float = 0.3):
    """Generate descriptive HTML report."""

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    pattern_groups = ['pos_patterns', 'neg_patterns']

    # Extract seqlets
    patterns_data = extract_seqlet_data(modisco_h5py, pattern_groups)

    # Create visualizations
    logo_paths = create_comprehensive_logos(patterns_data, output_dir, trim_threshold)
    distribution_paths = create_distribution_plots(patterns_data, output_dir)
    examples_data = create_seqlet_example_logos(patterns_data, output_dir, n_examples)

    # Get TOMTOM matches if database provided
    tomtom_data = {}
    if meme_motif_db is not None:
        from pathlib import Path
        if ttl:
            tomtom_df = tomtomlite_dataframe(Path(modisco_h5py), Path(output_dir), Path(meme_motif_db) if meme_motif_db else None,
                pattern_groups=pattern_groups, top_n_matches=top_n_matches,
                trim_threshold=trim_threshold)
        else:
            tomtom_df = generate_tomtom_dataframe(Path(modisco_h5py), Path(output_dir), Path(meme_motif_db) if meme_motif_db else None,
                is_writing_tomtom_matrix=False, pattern_groups=pattern_groups,
                top_n_matches=top_n_matches, trim_threshold=trim_threshold)

        # Convert to dictionary format
        for i, (pattern_tag, _) in enumerate(patterns_data.items()):
            if i < len(tomtom_df):
                tomtom_data[pattern_tag] = {}
                for j in range(top_n_matches):
                    match_col = f'match{j}'
                    pval_col = f'pval{j}' if ttl else f'qval{j}'
                    if match_col in tomtom_df.columns:
                        tomtom_data[pattern_tag][f'match_{j}'] = tomtom_df.iloc[i][match_col]
                        tomtom_data[pattern_tag][f'pval_{j}'] = tomtom_df.iloc[i][pval_col]

    # Generate HTML report
    template_str = (
        importlib.resources.files(templates).joinpath("descriptive_report.html").read_text()
    )
    template = Template(template_str)
    html_content = template.render(
        patterns_data=patterns_data,
        logo_paths=logo_paths,
        distribution_paths=distribution_paths,
        examples_data=examples_data,
        tomtom_data=tomtom_data,
        img_path_suffix=img_path_suffix,
        meme_motif_db=meme_motif_db,
        top_n_matches=top_n_matches,
        ttl=ttl
    )

    # Write HTML report
    report_path = os.path.join(output_dir, 'descriptive_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Descriptive report generated: {report_path}")
    return report_path