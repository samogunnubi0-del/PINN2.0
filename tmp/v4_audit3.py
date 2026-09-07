import pandas as pd, numpy as np, os, json
base = r'c:\Users\ogunn\Downloads\New folder\New folder\v4_transport_surrogate\results\geometry_rate_surrogate_pilot_v2_20260824'
d = pd.read_csv(os.path.join(base, 'geometry_rate_reference_cases.csv')).sort_values('family_id', kind='stable').reset_index(drop=True)
print('split counts:', dict(d.split.value_counts()))
print('refined_reference_role:', dict(d.refined_reference_role.value_counts(dropna=False)))
print()

test = d[d.split == 'test']
dev = d[d.split == 'development']
train = d[d.split == 'train']

print('=== A. THE DECISIVE CHECK: reference_k (train label) vs refined_k (eval label) ===')
for ch in ['n2n', 'ngamma']:
    ref = test['reference_k_%s_per_h_per_uA' % ch].to_numpy()
    rfn = test['refined_k_%s_per_h_per_uA' % ch].to_numpy()
    gap = np.abs(ref - rfn) / rfn
    print('  TEST %-6s  |reference-refined|/refined : median=%.4f%%  p95=%.4f%%  max=%.4f%%' % (ch, np.median(gap) * 100, np.percentile(gap, 95) * 100, gap.max() * 100))
    print('           signed mean bias (ref/refined - 1) = %+.4f%%' % ((ref / rfn - 1).mean() * 100))
print()
print('  --> Compare with reported MODEL errors on the same test rows:')
print('      GP   n2n median 0.526%  p95 1.991%   | ngamma median 3.034%  p95 3.984%')
print('      LSTM n2n median 0.588%  p95 2.736%   | ngamma median 2.919%  p95 4.129%')
print()

print('=== B. Would "predict reference_k exactly" already achieve the reported error? ===')
for ch, i in [('n2n', 0), ('ngamma', 1)]:
    ref = test['reference_k_%s_per_h_per_uA' % ch].to_numpy()
    rfn = test['refined_k_%s_per_h_per_uA' % ch].to_numpy()
    e = np.abs(ref - rfn) / rfn
    print('  ORACLE-on-training-label %-6s: median=%.4f%% p95=%.4f%%  (this is the FLOOR the model cannot beat)' % (ch, np.median(e) * 100, np.percentile(e, 95) * 100))
pooled = np.concatenate([np.abs(test['reference_k_n2n_per_h_per_uA'] - test['refined_k_n2n_per_h_per_uA']) / test['refined_k_n2n_per_h_per_uA'],
                        np.abs(test['reference_k_ngamma_per_h_per_uA'] - test['refined_k_ngamma_per_h_per_uA']) / test['refined_k_ngamma_per_h_per_uA']])
print('  ORACLE pooled(96): median=%.4f%% p95=%.4f%%   vs reported GP 1.971%%/3.791%%, LSTM 1.832%%/3.931%%' % (np.median(pooled) * 100, np.percentile(pooled, 95) * 100))
print()

print('=== C. How much correction must the model learn? point_k baseline vs refined_k truth ===')
for ch in ['n2n', 'ngamma']:
    pt = test['point_k_%s_per_h_per_uA' % ch].to_numpy()
    rfn = test['refined_k_%s_per_h_per_uA' % ch].to_numpy()
    r = pt / rfn
    print('  TEST %-6s point_k/refined_k : median=%.3f  min=%.3f max=%.3f  -> baseline-alone rel err median=%.1f%%' % (
        ch, np.median(r), r.min(), r.max(), np.median(np.abs(pt - rfn) / rfn) * 100))
print()

print('=== D. Reference wall-time (is any speedup claim supportable?) ===')
print(d[['reference_wall_time_s', 'refined_wall_time_s']].describe().to_string())
print('  train rows reference_wall_time_s first 6:', np.round(train.reference_wall_time_s.values[:6], 5))
print()

print('=== E. deuteron_energy is binary, not continuous ===')
print('  unique deuteron_energy_mev:', sorted(d.deuteron_energy_mev.unique()))
for c in ['target_distance_cm', 'beam_fwhm_mm', 'radial_offset_mm', 'target_radius_mm']:
    print('  %-20s range in data = [%.4f, %.4f]  declared bounds differ?' % (c, d[c].min(), d[c].max()))
print()

print('=== F. Does the test set actually cover the declared domain box? ===')
BOUNDS = {'target_distance_cm': (1.5, 3.0), 'beam_fwhm_mm': (1.0, 15.0), 'radial_offset_mm': (0.0, 8.0), 'target_radius_mm': (0.5, 5.0)}
for c, (lo, hi) in BOUNDS.items():
    print('  %-20s declared [%.2f, %.2f] ; TRAIN [%.3f, %.3f] ; TEST [%.3f, %.3f]' % (
        c, lo, hi, train[c].min(), train[c].max(), test[c].min(), test[c].max()))
print()

print('=== G. Reproduce the development -> selection/calibration split and check disjointness vs test ===')
dev_rows = np.flatnonzero((d.split == 'development').to_numpy())
order = np.random.default_rng(20260826).permutation(len(dev_rows))
sel = dev_rows[order[:len(order) // 2]]
cal = dev_rows[order[len(order) // 2:]]
test_rows = np.flatnonzero((d.split == 'test').to_numpy())
print('  n_sel=%d n_cal=%d n_test=%d' % (len(sel), len(cal), len(test_rows)))
print('  sel ^ cal overlap: %d ; sel ^ test: %d ; cal ^ test: %d' % (
    len(set(sel) & set(cal)), len(set(sel) & set(test_rows)), len(set(cal) & set(test_rows))))
cal_ids = sorted(d.loc[cal, 'family_id'].tolist())
test_ids = sorted(d.loc[test_rows, 'family_id'].tolist())
print('  calibration family_ids:', cal_ids)
print('  test family_ids (first 20):', test_ids[:20])
print('  --> family_id overlap calibration/test:', sorted(set(cal_ids) & set(test_ids)))
