
XL = [-8.435281877979797 ,-134.0391035421266 ,-31.677134243111425]
# PLYファイルに書き込む
header = f"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
with open("seal_point.ply", "w") as ply_file:
    ply_file.write(header)
    for vertex in range(1):
        vertex = list(map(str, XL[:]))
        ply_file.write(" ".join(vertex) + "\n")