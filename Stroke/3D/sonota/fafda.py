import numpy as np

select_corners = [np.array([[[1123.,  777.],
                             [1157.,  779.],
                             [1158.,  813.],
                             [1123.,  811.]]], dtype=np.float32),
                  np.array([[[1122.,  737.],
                             [1157.,  739.],
                             [1157.,  774.],
                             [1123.,  772.]]], dtype=np.float32),
                  np.array([[[1162.,  739.],
                             [1198.,  741.],
                             [1198.,  776.],
                             [1162.,  774.]]], dtype=np.float32),
                  np.array([[[1162.,  779.],
                             [1198.,  782.],
                             [1199.,  816.],
                             [1163.,  814.]]], dtype=np.float32)]

# 必要な形状に変換
result = np.array([corner[0] for corner in select_corners])

print(result)
