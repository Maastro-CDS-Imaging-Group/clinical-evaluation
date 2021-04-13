# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/baseline_2d/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/baseline_2d 
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/baseline_2d/ --output_dir /data/media_results/test_analysis/baseline_2d --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/3d_vnet/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/3d_vnet
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/3d_vnet/ --output_dir /data/media_results/test_analysis/3d_vnet --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/revvnet/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/revvnet
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/revvnet/ --output_dir /data/media_results/test_analysis/revvnet --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/ssim/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/ssim
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/ssim/ --output_dir /data/media_results/test_analysis/ssim --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/ssim_med_patch/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/ssim_med_patch
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/ssim_med_patch/ --output_dir /data/media_results/test_analysis/ssim_med_patch --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/ssim_small_patch/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/ssim_small_patch
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/ssim_small_patch/ --output_dir /data/media_results/test_analysis/ssim_small_patch --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

# python projects/nki_cervix_cbct_ct/utils/restructure_dirs_for_eval.py /data/test /data/media_results/translated/ssim_amp/saved/CBCTtoCTVal --output_dir /data/media_results/test_evaluation/ssim_amp
python projects/nki_cervix_cbct_ct/evaluation/evaluate_test_set.py /data/media_results/test_evaluation/ssim_amp --output_dir /data/media_results/test_analysis/ssim_amp --masks BODY BLADDER BOWELAREA RECTUM SIGMOID --cores 8 -v

