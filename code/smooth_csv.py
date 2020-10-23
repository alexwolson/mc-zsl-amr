import pandas

filenames = ["run_large_lg_lr_001_opt_rms_bs_32_VGG16_tag_val_acc.csv","run_medium_med_lr_001_opt_rms_bs_64_VGG16_tag_val_acc.csv"]#["run_adagrad_tag_val_acc.csv","run_rmsproplr001_tag_val_acc.csv","run_adam_tag_val_acc.csv","run_rmsproplr01_tag_val_acc.csv","run_bs256_tag_val_acc.csv","run_rmsproplr1_tag_val_acc.csv","run_bs32_tag_val_acc.csv","run_bs64_tag_val_acc.csv","run_lstm3layers_no_drop_tag_val_acc.csv","run_lstm5layers_no_drop_tag_val_acc.csv","run_rmsproplr0001_tag_val_acc.csv"]
for name in filenames:
    c = pandas.read_csv(name)
    c['Value'] = pandas.ewma(c['Value'], com=5)
    c.to_csv(name)