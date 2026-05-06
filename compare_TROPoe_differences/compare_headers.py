def compare_headers(file1, file2, label1="File 1", label2="File 2"):
    with open(file1, "r") as f:
        lines1 = set(f.read().splitlines())
    with open(file2, "r") as f:
        lines2 = set(f.read().splitlines())

    only_in_1 = lines1 - lines2
    only_in_2 = lines2 - lines1

    if not only_in_1 and not only_in_2:
        print("Headers are identical!")
    else:
        if only_in_1:
            print(f"\n=== Only in {label1} ===")
            for line in sorted(only_in_1):
                print(f"  {line}")
        if only_in_2:
            print(f"\n=== Only in {label2} ===")
            for line in sorted(only_in_2):
                print(f"  {line}")

compare_headers("my_header.txt", "dave_header.txt", label1="Yours", label2="Dave's")

print('end')