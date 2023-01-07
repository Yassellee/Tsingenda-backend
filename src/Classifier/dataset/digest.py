with open("resume_train.txt", "r") as f:
    lines = f.readlines()
    # add labels to the end of each line, separated by tab
    # 1-20 are positive, 21-40 are negative
    # 41-61 are positive, 62-82 are negative
    new_lines = f.readlines()

with open("resume_train.txt", "w") as f:
    f.write(new_lines[0]+'\t'+'0'+'\r\n')
    f.write(new_lines[1]+'\t'+'1'+'\r\n')

with open("train.txt", "w") as f:
    for line in new_lines:
        f.write(line+'\r\n')

# import numpy

# with open("train.txt", "r") as f:
#     lines = f.readlines()

# with open("dev.txt", "w") as f:
#     # randomly choose 20 sentences from lines and write to dev.txt
#     dev_lines = numpy.random.choice(lines, 20, replace=False)
#     for line in dev_lines:
#         f.write(line)
