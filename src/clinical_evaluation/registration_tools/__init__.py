import pandas as pd


class RegistrationInformation:

    def __init__(self, outdir='out', save_to='registration_info.csv'):
        self.outdir = outdir
        self.save_to = save_to
        self.df = pd.DataFrame()

    def add_info(self, info_dict):
        self.df = self.df.append(info_dict, ignore_index=True)
        self.save_info()

    def save_info(self):
        if not self.df.empty:
            csv_path = self.outdir / self.save_to
            self.df.to_csv(csv_path)
            
    def get_aggregate_dataframe(self):
        self.mean_df = self.df.mean(axis=0, skipna=True)
        return self.mean_df
