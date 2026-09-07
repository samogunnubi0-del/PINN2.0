import pandas as pd, numpy as np, os, json, hashlib
os.chdir(r'c:\Users\ogunn\Downloads\New folder\tmp\v4_extract\v4_gp_lstm_comparison_c3a56472508c')
gp = pd.read_csv('gp/test_predictions.csv'); ls = pd.read_csv('lstm/test_predictions.csv')
gpr = json.load(open('gp/model_result.json')); lsr = json.load(open('lstm/model_result.json'))
gpd_ = pd.read_csv('gp/domain_challenges.csv'); lsd = pd.read_csv('lstm/domain_challenges.csv')
paired = pd.read_csv('combined/paired_case_comparison.csv')
comp = json.load(open('combined/final_model_comparison.json'))

print('=== 7. domain_challenges composition and rejection mechanism ===')
for name, d in [('GP', gpd_), ('LSTM', lsd)]:
    print('   %s rows=%d' % (name, len(d)))
    print(d.groupby(['challenge_kind', 'challenge_label', 'domain_accepted', 'domain_reason']).size().to_string())
    print('   --- reject rate by kind ---')
    print(d.groupby('challenge_kind').domain_accepted.agg(['count', 'sum', lambda s: 1 - s.mean()]).to_string())
    print('   nearest_training_distance by kind:')
    print(d.groupby('challenge_kind').nearest_training_distance.describe()[['min', '50%', 'max']].to_string())
    print()

print('=== 7b. Are ANY explicit-OOD cases rejected by the DISTANCE rule rather than box bounds? ===')
thr = gpr['domain_guard']['nearest_training_distance_threshold']
ood = gpd_[gpd_.challenge_kind == 'explicit_out_of_range']
print('   threshold=%.4f ; OOD cases with distance > threshold: %d / %d' % (thr, (ood.nearest_training_distance > thr).sum(), len(ood)))
print('   all OOD domain_reason values: %s' % dict(ood.domain_reason.value_counts()))
sc = gpd_[gpd_.challenge_kind == 'in_range_sparse_corner']
print('   sparse corners: n=%d, rejected=%d, distance>thr=%d' % (len(sc), (~sc.domain_accepted).sum(), (sc.nearest_training_distance > thr).sum()))
print('   sparse corner reasons: %s' % dict(sc.domain_reason.value_counts()))

print()
print('=== 8. LSTM ensemble SD floor check ===')
floor = lsr['training']['selection_residual_floor_log']
print('   selection_residual_floor_log = %s' % floor)
for ch, f in [('n2n', floor[0]), ('ngamma', floor[1])]:
    col = ls['log_prediction_std_' + ch]
    print('   LSTM log_prediction_std_%-6s: min=%.6f med=%.6f max=%.6f | floor=%.6f | frac at/below floor+1e-9 = %.3f' % (
        ch, col.min(), col.median(), col.max(), f, (col <= f + 1e-9).mean()))
    print('        frac within 1%% of floor = %.3f ; std of column = %.6g' % ((np.abs(col - f) / f < 0.01).mean(), col.std()))
print('   GP log_prediction_std_n2n:    min=%.6f med=%.6f max=%.6f std=%.4g' % (gp.log_prediction_std_n2n.min(), gp.log_prediction_std_n2n.median(), gp.log_prediction_std_n2n.max(), gp.log_prediction_std_n2n.std()))
print('   GP log_prediction_std_ngamma: min=%.6f med=%.6f max=%.6f std=%.4g' % (gp.log_prediction_std_ngamma.min(), gp.log_prediction_std_ngamma.median(), gp.log_prediction_std_ngamma.max(), gp.log_prediction_std_ngamma.std()))
print('   LSTM OOD std ratio (json) = %.4f  ; GP = %.4f' % (
    lsr['domain_challenges']['explicit_ood_to_locked_uncertainty_ratio'], gpr['domain_challenges']['explicit_ood_to_locked_uncertainty_ratio']))

print()
print('=== 9. Figure 1 panel (c): plotted quantity vs annotated numbers ===')
print('   PLOTTED ECDF = paired.gp_case_mean_relative_error : median=%.4f%% p95=%.4f%%' % (
    np.median(paired.gp_case_mean_relative_error) * 100, np.percentile(paired.gp_case_mean_relative_error, 95) * 100))
print('   ANNOTATED    = comparison[gp][median/p95]         : median=%.4f%% p95=%.4f%%' % (
    comp['gp']['median_relative_error'] * 100, comp['gp']['p95_relative_error'] * 100))
print('   PLOTTED ECDF lstm: median=%.4f%% p95=%.4f%%   ANNOTATED lstm: median=%.4f%% p95=%.4f%%' % (
    np.median(paired.lstm_case_mean_relative_error) * 100, np.percentile(paired.lstm_case_mean_relative_error, 95) * 100,
    comp['lstm']['median_relative_error'] * 100, comp['lstm']['p95_relative_error'] * 100))

print()
print('=== 10. Figure 1 panel (d): plotted deltas vs annotated bootstrap statistic ===')
d = paired.lstm_minus_gp_case_error
print('   plotted delta (case-mean): mean=%+.4f pp median=%+.4f pp min=%+.4f max=%+.4f ; n_lstm_better=%d/48' % (
    d.mean() * 100, d.median() * 100, d.min() * 100, d.max() * 100, (d < 0).sum()))
print('   annotated observed_difference = %+.4f pp  (statistic = %s)' % (
    comp['paired_bootstrap']['observed_difference'] * 100, comp['paired_bootstrap']['statistic']))
print('   annotated CI = [%+.4f, %+.4f] pp' % tuple(v * 100 for v in comp['paired_bootstrap']['confidence_interval_95']))

print()
print('=== 11. Provenance hashes: are committed figures stale? ===')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
man = json.load(open(r'c:\Users\ogunn\Downloads\New folder\results\v4_publication_figures\figure_manifest.json'))
gen = r'c:\Users\ogunn\Downloads\New folder\analysis\plot_v4_publication_figures.py'
zp = r'C:\Users\ogunn\Downloads\V4_GP_LSTM_FINAL_c3a56472508c.zip'
print('   generator sha now  = %s' % sha(gen))
print('   generator sha manif= %s   MATCH=%s' % (man['generator_sha256'], sha(gen) == man['generator_sha256']))
print('   zip sha now        = %s' % sha(zp))
print('   zip sha manifest   = %s   MATCH=%s' % (man['input_zip_sha256'], sha(zp) == man['input_zip_sha256']))
outdir = r'c:\Users\ogunn\Downloads\New folder\results\v4_publication_figures'
for k, v in man['outputs'].items():
    p = os.path.join(outdir, k)
    ok = os.path.exists(p) and sha(p) == v
    print('   output %-46s MATCH=%s' % (k, ok))

print()
print('=== 12. Does the ZIP dataset_sha256 correspond to any file in the repo? ===')
print('   dataset_sha256 = %s' % comp['dataset_sha256'])
print('   (dataset CSV itself is NOT in the ZIP; ZIP members:)')
import zipfile
print('   %s' % [n for n in zipfile.ZipFile(zp).namelist() if 'reference_cases' in n or 'dataset' in n])
