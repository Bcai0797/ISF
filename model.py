import torch
from torch import nn

def fc_layer(in_features, out_features, activation, flag_norm=True, norm_type='bn'):
    module_list = [nn.Linear(in_features, out_features)]
    if flag_norm:
        if norm_type == 'bn':
            module_list.append(nn.BatchNorm1d(out_features))
        else:            
            module_list.append(nn.LayerNorm(out_features))
    if activation is not None:
        module_list.append(activation())
    return nn.Sequential(*module_list)


def positional_encoding(pos, d_model):
    n_pos = pos.size(0)
    pos = pos.unsqueeze(-1)
    params = 1e4 ** (-torch.arange(d_model // 2) / d_model).to(pos.device)
    params = pos * params.unsqueeze(0)
    sin_array = torch.sin(params).unsqueeze(-1)
    cos_array = torch.cos(params).unsqueeze(-1)
    return torch.cat([sin_array, cos_array], dim=-1).reshape(n_pos, -1)


class ISF(nn.Module):
    def __init__(self, in_size, base_units, n_embed, embed_dim, t_min, t_max, flag_embed=True, flag_bn=True, dropout_p=0.5) -> None:
        super(ISF, self).__init__()
        self.flag_embed = flag_embed
        self.embed = lambda x: x
        if self.flag_embed:
            self.embed = nn.Embedding(n_embed, embed_dim)
            in_size *= embed_dim
        self.relu = nn.ReLU
        self.gelu = nn.GELU
        self.sigmoid = nn.Sigmoid

        self.fc1 = fc_layer(in_size, base_units*2, self.gelu, flag_bn, 'ln')
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = fc_layer(base_units*2, base_units*4, self.gelu, flag_bn, 'ln')
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = fc_layer(base_units*4, base_units*2, self.gelu, flag_bn, 'ln')
        self.dropout3 = nn.Dropout(dropout_p)

        self.fc4 = fc_layer(base_units*2, base_units*2, self.gelu, True, 'ln')
        self.fc5 = fc_layer(base_units*2, 1, self.sigmoid, False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x):
        n_batch = x.size(0)
        x = self.embed(x).reshape(n_batch, -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        return x

    def predict_h_z_t(self, z, t):
        t_encode = positional_encoding(t, z.size(-1))
        x = z + t_encode
        x = self.fc4(x)
        x = self.fc5(x)

        return x

    def forward(self, x, t):
        z = self.encode(x)
        n_batch = z.size(0)
        t_arr = t.expand(n_batch, -1)
        h = self.predict_h_z_t(z, t_arr)
        return h
    
    def predict_h_x_t_list(self, x, t_list):
        z = self.encode(x)
        return self.predict_h_z_t_list(z, t_list)

    def predict_h_z_t_list(self, z, t_list):
        n_batch, n_t = z.size(0), t_list.size(0)
        z = z.reshape(n_batch, 1, -1).expand(n_batch, n_t, -1).reshape(n_batch*n_t, -1)
        t_list = t_list.reshape(1, n_t, -1).expand(n_batch, n_t, -1).reshape(n_batch*n_t, -1)

        h = self.predict_h_z_t(z, t_list).reshape(n_batch, n_t)
        return h

    def predict_ln_1_h(self, h):
        return torch.log(torch.clip(1-h, 1e-9, 1))

    def predict_ln_1_h_z_t_list(self, z, t_list):
        return self.predict_ln_1_h(self.predict_h_z_t_list(z, t_list))

    def s_simpson(self, x, t_min, t_max, t_step=1):
        t_minus_arr = torch.arange(t_min, t_max-2*t_step, t_step).to(x.device)
        t_plus_arr = t_minus_arr + t_step
        t_mean_arr = (t_minus_arr + t_plus_arr) / 2

        z = self.encode(x)
        n_batch = z.size(0)

        h_minus_arr = self.predict_ln_1_h_z_t_list(z, t_minus_arr)
        h_plus_arr = self.predict_ln_1_h_z_t_list(z, t_plus_arr)
        h_mean_arr = self.predict_ln_1_h_z_t_list(z, t_mean_arr)
        h_arr = (h_minus_arr + h_plus_arr + h_mean_arr * 4) / 6 * t_step # B * T
        s_arr = torch.exp(torch.cumsum(h_arr, dim=-1)) # B * T
        s_arr = torch.concat([torch.ones(n_batch, 1).to(s_arr.device), s_arr, torch.zeros(n_batch, 1).to(s_arr.device)], dim=-1)
        return s_arr

    def p_simpson(self, x, t_min, t_max, t_step=1):
        s_arr = self.s_simpson(x, t_min, t_max, t_step)
        p_arr = -torch.diff(s_arr, dim=-1)
        return p_arr


class ISF_concat(nn.Module):
    def __init__(self, in_size, base_units, n_embed, embed_dim, t_min, t_max, flag_embed=True, flag_bn=True, dropout_p=0.5) -> None:
        super(ISF_concat, self).__init__()
        self.flag_embed = flag_embed
        self.embed = lambda x: x
        if self.flag_embed:
            self.embed = nn.Embedding(n_embed, embed_dim)
            in_size *= embed_dim
        self.relu = nn.ReLU
        self.gelu = nn.GELU
        self.sigmoid = nn.Sigmoid

        self.fc1 = fc_layer(in_size, base_units*2, self.gelu, flag_bn, 'ln')
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = fc_layer(base_units*2, base_units*4, self.gelu, flag_bn, 'ln')
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = fc_layer(base_units*4, base_units*2, self.gelu, flag_bn, 'ln')
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = fc_layer(base_units*2+1, base_units*2, self.gelu, True, 'ln')
        self.fc5 = fc_layer(base_units*2, 1, self.sigmoid, False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x):
        n_batch = x.size(0)
        x = self.embed(x).reshape(n_batch, -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        return x

    def predict_h_z_t(self, z, t):
        x = torch.concat([z, t], dim=-1)
        x = self.fc4(x)

        x = self.fc5(x)

        return x

    def forward(self, x, t):
        z = self.encode(x)
        n_batch = z.size(0)
        t_arr = t.expand(n_batch, -1)
        h = self.predict_h_z_t(z, t_arr)
        return h
    
    def predict_h_x_t_list(self, x, t_list):
        z = self.encode(x)
        return self.predict_h_z_t_list(z, t_list)

    def predict_h_z_t_list(self, z, t_list):
        n_batch, n_t = z.size(0), t_list.size(0)
        z = z.reshape(n_batch, 1, -1).expand(n_batch, n_t, -1).reshape(n_batch*n_t, -1)
        t_list = t_list.reshape(1, n_t, -1).expand(n_batch, n_t, -1).reshape(n_batch*n_t, -1)

        h = self.predict_h_z_t(z, t_list).reshape(n_batch, n_t)
        return h

    def predict_ln_1_h(self, h):
        return torch.log(torch.clip(1-h, 1e-9, 1))

    def predict_ln_1_h_z_t_list(self, z, t_list):
        return self.predict_ln_1_h(self.predict_h_z_t_list(z, t_list))

    def s_simpson(self, x, t_min, t_max, t_step=1):
        t_minus_arr = torch.arange(t_min, t_max-2*t_step, t_step).to(x.device)
        t_plus_arr = t_minus_arr + t_step
        t_mean_arr = (t_minus_arr + t_plus_arr) / 2

        z = self.encode(x)
        n_batch = z.size(0)

        h_minus_arr = self.predict_ln_1_h_z_t_list(z, t_minus_arr)
        h_plus_arr = self.predict_ln_1_h_z_t_list(z, t_plus_arr)
        h_mean_arr = self.predict_ln_1_h_z_t_list(z, t_mean_arr)
        h_arr = (h_minus_arr + h_plus_arr + h_mean_arr * 4) / 6 * t_step # B * T
        s_arr = torch.exp(torch.cumsum(h_arr, dim=-1)) # B * T
        s_arr = torch.concat([torch.ones(n_batch, 1).to(s_arr.device), s_arr, torch.zeros(n_batch, 1).to(s_arr.device)], dim=-1)
        return s_arr

    def p_simpson(self, x, t_min, t_max, t_step=1):
        s_arr = self.s_simpson(x, t_min, t_max, t_step)
        p_arr = -torch.diff(s_arr, dim=-1)
        return p_arr


if __name__ == '__main__':
    model = ISF(10, 10, 10, 10, 0, 10, False)
    model.eval()
    a_test = model.forward(torch.ones(2, 10), torch.ones(1)*3)
    print(a_test)
    t_test = model.predict_h_x_t_list(torch.concat([torch.ones(2, 10), torch.zeros(1, 10)], dim=0), torch.arange(5)*1.0)
    print(t_test)
    b_test = model.p_simpson(torch.ones(2, 10), 1,  4, 1)
    print(b_test)