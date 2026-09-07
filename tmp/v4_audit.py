import pandas as pd, numpy as np, os, json
os.chdir(r'c:\Users\ogunn\Downloads\New folder\tmp\v4_extract\v4_gp_lstm_comparison_c3a56472508c')
gp = pd.read_csv('gp/test_predictions.csv')
ls = pd.read_csv('lstm/test_predictions.csv')
gpr = json.load(open('gp/model_result.json'))
lsr = json.load(open('lstm/model_result.json'))
gs = pd.read_csv('gp/ac225_schedule_predictions.csv')
lss = pd.read_csv('lstm/ac225_schedule_predictions.csv')
MODELS = [('GP', gp, gpr, gs), ('LSTM', ls, lsr, lss)]

print('=== 1. Is Ac-225 inventory relative error identical to (n,2n) rate relative error? ===')
for name, tp, res, sc in MODELS:
    m = sc.merge(tp[['family_id', 'relative_error_n2n']], on='family_id')
    for sch in ['short', 'medium', 'long']:
        s = m[m.schedule == sch]
        d = np.abs(s.relative_error.values - s.relative_error_n2n.values).max()
        print('   %s %-7s n=%d  max|ac225_relerr - n2n_relerr| = %.3e' % (name, sch, len(s), d))
    piv = m.pivot_table(index='family_id', columns='schedule', values='relative_error')
    print('   %s max spread across the 3 schedules = %.3e' % (name, (piv.max(axis=1) - piv.min(axis=1)).max()))
    print('   %s ac225 truth ratio long/short (should differ if schedules differ): %s' % (
        name, np.round((sc[sc.schedule == 'long'].truth_ac225_norm.values / sc[sc.schedule == 'short'].truth_ac225_norm.values)[:4], 4)))

print()
print('=== 2. Reproduce reported test metrics from CSV ===')
for name, tp, res, sc in MODELS:
    pooled = np.concatenate([tp.relative_error_n2n.values, tp.relative_error_ngamma.values])
    tm = res['test_metrics']
    print('   %s pooled(96) median = %.12f | json median_relative_error = %.12f' % (name, np.median(pooled), tm['median_relative_error']))
    print('   %s pooled(96) p95    = %.12f | json p95_relative_error    = %.12f' % (name, np.percentile(pooled, 95), tm['p95_relative_error']))
    print('   %s n2n median=%.12f (json %.12f) ; ngamma median=%.12f (json %.12f)' % (
        name, np.median(tp.relative_error_n2n), tm['median_n2n_relative_error'],
        np.median(tp.relative_error_ngamma), tm['median_ngamma_relative_error']))
    print('   %s casemean median=%.12f (json %.12f) p95=%.12f (json %.12f)' % (
        name, np.median(tp.case_mean_relative_error), tm['median_case_mean_relative_error'],
        np.percentile(tp.case_mean_relative_error, 95), tm['p95_case_mean_relative_error']))
    print('   %s ac225 grid median=%.12f (json %.12f) ; grid p95=%.12f (json %.12f)' % (
        name, np.median(sc.relative_error), res['ac225_inventory']['grid_median_relative_error'],
        np.percentile(sc.relative_error, 95), res['ac225_inventory']['grid_p95_relative_error']))

print()
print('=== 3. boundary_case subset vs full locked test ===')
for name, tp, res, sc in MODELS:
    b = tp[tp.boundary_case]
    pb = np.concatenate([b.relative_error_n2n.values, b.relative_error_ngamma.values])
    pa = np.concatenate([tp.relative_error_n2n.values, tp.relative_error_ngamma.values])
    bm = res['boundary_test']['metrics']
    print('   %s n_boundary=%d of %d (json count=%d)' % (name, len(b), len(tp), res['boundary_test']['count']))
    print('      boundary pooled median = %.15f' % np.median(pb))
    print('      FULL     pooled median = %.15f' % np.median(pa))
    print('      json boundary   median = %.15f' % bm['median_relative_error'])
    print('      boundary worst=%.15f  full worst=%.15f  json boundary worst=%.15f' % (pb.max(), pa.max(), bm['worst_relative_error']))
    print('      boundary p95=%.12f json=%.12f' % (np.percentile(pb, 95), bm['p95_relative_error']))

print()
print('=== 4. Coverage recomputation (interval columns vs reported coverage) ===')
for name, tp, res, sc in MODELS:
    for tag, lo_n, hi_n, lo_g, hi_g in [
        ('raw_two_standard_deviations', 'raw_2sd_lower_n2n', 'raw_2sd_upper_n2n', 'raw_2sd_lower_ngamma', 'raw_2sd_upper_ngamma'),
        ('calibrated_95', 'calibrated_95_lower_n2n', 'calibrated_95_upper_n2n', 'calibrated_95_lower_ngamma', 'calibrated_95_upper_ngamma')]:
        cn = ((tp.truth_n2n_per_h_per_uA >= tp[lo_n]) & (tp.truth_n2n_per_h_per_uA <= tp[hi_n]))
        cg = ((tp.truth_ngamma_per_h_per_uA >= tp[lo_g]) & (tp.truth_ngamma_per_h_per_uA <= tp[hi_g]))
        joint = (cn & cg).mean()
        j = res['uncertainty'][tag]
        print('   %s %-28s joint=%.6f (json %.6f) marg=[%.4f,%.4f] (json [%.4f,%.4f])' % (
            name, tag, joint, j['joint_case_coverage'], cn.mean(), cg.mean(), j['marginal_coverage'][0], j['marginal_coverage'][1]))

print()
print('=== 5. Domain guard on locked test: any rejection? distance vs threshold ===')
for name, tp, res, sc in MODELS:
    thr = res['domain_guard']['nearest_training_distance_threshold']
    print('   %s threshold=%.6f  test dist min=%.4f max=%.4f  n_accepted=%d/%d  reasons=%s' % (
        name, thr, tp.nearest_training_distance.min(), tp.nearest_training_distance.max(),
        int(tp.domain_accepted.sum()), len(tp), dict(tp.domain_reason.value_counts())))

print()
print('=== 6. Conformal multiplier check: calibrated width / raw width ===')
for name, tp, res, sc in MODELS:
    u = res['uncertainty']
    mult = u['joint_conformal_multiplier']
    w_raw_n = np.log(tp.raw_2sd_upper_n2n / tp.raw_2sd_lower_n2n)
    w_cal_n = np.log(tp.calibrated_95_upper_n2n / tp.calibrated_95_lower_n2n)
    print('   %s conformal multiplier=%.6f  median(cal_width/raw_width) n2n=%.6f (expect mult/2=%.6f)' % (
        name, mult, np.median(w_cal_n / w_raw_n), mult / 2.0))
    print('       median log_prediction_std_n2n=%.6f  ngamma=%.6f' % (
        tp.log_prediction_std_n2n.median(), tp.log_prediction_std_ngamma.median()))
