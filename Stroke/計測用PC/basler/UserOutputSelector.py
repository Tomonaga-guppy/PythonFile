# マスターカメラのUserOutputSelectorで使える値を確認
available_outputs = master_camera.UserOutputSelector.GetEntries()

# 利用可能なオプションを表示
for entry in available_outputs:
    print(entry.GetSymbolic())
