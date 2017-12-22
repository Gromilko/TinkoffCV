lines = []
with open('submit.csv', 'r') as f:
    lines = f.readlines()

orig = ['5,54', '61,37', '100,4', '101,93', '146,58',
        '171,2', '172,2', '174,2', '177,13', '178,0',
        '179,2', '180,2', '181,2', '182,7', '183,7']
fix = ['5,57', '61,84', '100,80', '101,80', '146,70',
       '171,17', '172,17', '174,17', '177,15', '178,17',
       '179,15', '180,17', '181,17', '182,17', '183,17']

with open('submit_fix.csv', 'w') as f:
    for i in range(len(lines)):
        # игнорируем \n в конце считанных строк
        if lines[i][:-1] in orig:
            a = orig.index(lines[i][:-1])
            f.write(fix[a]+'\n')
        else:
            f.write(lines[i])