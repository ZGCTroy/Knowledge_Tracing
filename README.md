#Train Val Test
Train : Val : Test = 56:14:30
# DKT AssisTments2009


LSTM_hidden = 100 , num_LSTM = 1, input_embedding_dim = 100, 


batch_size = 64, epoch = 10


| Dataset          | user id  |  problem id | skill id |
|------------------|----------|-------------|----------|
| ASSISTments2009  | 1 ~ 4151 | 1 ~ 16891   |  1 ~ 111 |



| Dataset          | ACC (%) | AUC (%) | version |Hyper Parameters |
|------------------|-----|-----|-----|-------------|
| ASSISTments2009  | 77.02 ± 0.07 | 81.81 ± 0.10 | other's| input_dim=100, hidden_dim=100 |
| ASSISTments2009  | 78 | 84 | pre_200 | input_dim=100, hidden_dim=100 |
| ASSISTments2009  | 77 | 83 | post_200 |input_dim=100, hidden_dim=100 |