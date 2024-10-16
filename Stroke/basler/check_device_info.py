import pypylon.pylon as py

tlf = py.TlFactory.GetInstance()
# list first five available cameras
for d in tlf.EnumerateDevices()[:5]:
    print("----")
    # typical filter rules:
    print(d.GetDeviceClass())
    print(d.GetModelName())
    print(d.GetSerialNumber())
    print(d.GetUserDefinedName())
    # other filter rules
    print(d.GetFullName())
    print(d.GetFriendlyName())
