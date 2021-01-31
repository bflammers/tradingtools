try:
    from ..datautils import HTTPLoader, ThreadStream
except:
    from tradingtools.data.datautils import HTTPLoader, ThreadStream


class SentiCryptLoader(ThreadStream):

    api = HTTPLoader(url="https://api.senticrypt.com/v1/bitcoin.json")

    def __init__(self, interval_time=30) -> None:
        super().__init__(lifo=True)
        self.add_producer(producer=self.api.make_request, interval_time=interval_time)


if __name__ == "__main__":

    from time import sleep

    stream = SentiCryptLoader(interval_time=1)

    for i in range(100):
        x = stream.get_next(return_empty=False)
        try:
            print(x[-1])
        except:
            print(x)
        sleep(0.1)