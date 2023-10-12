    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, n, audio_ft_size, n_tokens, seq_len):
            self.audio_ft_size = audio_ft_size
            self.n_tokens = n_tokens
            self.seq_len = seq_len
            self.n = n
            pass

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "audio_fts": torch.randn(self.seq_len, self.audio_ft_size),
                "chart_tokens": torch.randint(0, self.n_tokens, (self.seq_len,)),
            }
