import torch

REQUEST_GRAPH_DELETION = True  # if True, the error always occurs, if False, only sometimes
TEST_REPEAT_COUNT = 10000  # 200


def run_loop():
    x = torch.ones((32, 64, 64), requires_grad=True)

    while x.shape[0] > 0:
        yield x.shape  # for printing

        if REQUEST_GRAPH_DELETION:
            del out  # time.sleep(0.0001) helps, but shoudn't be relied on

        x.detach_().requires_grad_()

        out = x.sum()
        out.backward()  # incorrect gradient shape because some metadata is not updated

        with torch.no_grad():  # change the shape
            N = x.shape[0] // 2

            grad = x.grad
            x.set_(x.data[:N])
            x.grad = grad[:N]


for shape in run_loop():
    print(shape)

if TEST_REPEAT_COUNT > 0:  # repeats
    error_count, succ_count = 0, 0
    for i in range(TEST_REPEAT_COUNT):
        try:
            shapes = []
            for shape in run_loop():
                shapes.append(shape[0])
            print("OK   ", shapes)
            succ_count += 1
        except:
            print("ERROR", shapes)
            error_count += 1
    print(f"{error_count} errors, {succ_count} successes")
