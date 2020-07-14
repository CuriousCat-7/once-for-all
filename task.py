from speed_task import *
from ofa.model_zoo import ofa_net
import random
from typing import List
import yaml


class OFATask(Task):
    def __init__(
            self,
            net="ofa_mbv3_d234_e346_k357_w1.0",
            seed=123,
            num_sample=3,
            run_times=10,
            ptype="DSP",
            log_level="INFO"):
        """
        对ofa进行测速，
        其中将Hsigmoid, Hswish 分别替换成sigmoid 和 swish,
        因为:
        一来SNPE对其的图重构会出难以名状的bug，
        二来在SNPE上，Hard 竟没有原版的速度快
        """
        self.name = f"{net}_seed{seed}_num_sample{num_sample}_run_times{run_times}_ptype{ptype}"
        self.ofa_network = ofa_net(net, False)  # 只测速不测精度
        random.seed(seed)
        self.img_sizes = list(range(128, 225, 4))
        self.num_sample = num_sample
        self.run_times = run_times
        self.ptype = ptype
        logger.remove()
        logger.add(sys.stderr, level=log_level)


    def cfg2subnet(self, cfg)->nn.Module:
        self.ofa_network.set_active_subnet_with_cfg(cfg)
        subnet = self.ofa_network.get_active_subnet()
        assert cfg == subnet.numeric_cfg
        return subnet

    def measure_tasks(self, numeric_cfgs:List[dict]) -> pd.DataFrame:
        # hyper params
        run_times = self.run_times
        opset_version = 10
        ptype = self.ptype

        ks = [[] for i in range(20)]
        k_names = [f"k_{i}" for i in range(20)]
        es = [[] for i in range(20)]
        e_names = [f"e_{i}" for i in range(20)]
        ds = [[] for i in range(5)]
        d_names = [f"d_{i}" for i in range(5)]
        Hs = []
        Ws = []
        times = []
        flopses = []
        params = []
        for cfg in tqdm(numeric_cfgs):
            subnet = self.cfg2subnet(cfg)
            input_size = [3, cfg["wid"][0], cfg["wid"][1]]
            flops, param = get_model_complexity_info(
                    subnet, input_size, False, False)
            flopses.append(flops)
            params.append(param)
            for i in range(20):
                ks[i].append(cfg["ks"][i])
                es[i].append(cfg["e"][i])
            for i in range(5):
                ds[i].append(cfg["d"][i])
            Hs.append(cfg["wid"][0])
            Ws.append(cfg["wid"][1])
            try:
                client = LatencyMeasureClient(
                    opset_version=opset_version,
                    run_times=run_times,
                    ptype=ptype)
                time = client.model2latency(subnet, input_size)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning("task {} failed as {}", cfg, e)
                time = math.nan
            times.append(time)

        rdict = {}
        rdict["H"] = Hs
        rdict["W"] = Ws
        rdict["time"] = times
        rdict["flops"] = flopses
        rdict["param"] = params
        for k_name, k in zip(k_names, ks):
            rdict[k_name] = k
        for e_name, e in zip(e_names, es):
            rdict[e_name] = e
        for d_name, d in zip(d_names, ds):
            rdict[d_name] = d

        df = pd.DataFrame(rdict)
        return df

    def run(self):
        df = pd.DataFrame()
        while len(df) < self.num_sample:
            remain = self.num_sample - len(df)
            batch_size = min(5, remain)
            numeric_cfgs = self.get_batch_cfg(batch_size)
            new_df = self.measure_tasks(numeric_cfgs)
            df = pd.concat([new_df, df])
            save_path = f'{self.ROOT}/{self.name}.pkl'
            df.to_pickle(save_path)
            logger.info("have {} / {}, save to {}", len(df), self.num_sample, save_path)
        print(df)
        return df

    def get_batch_cfg(self, num=1):
        numeric_cfgs = []
        for i in range(num):
            cfg = self.ofa_network.sample_active_subnet(with_set=False)
            cfg["wid"] = [random.choice(self.img_sizes), random.choice(self.img_sizes)]  # H, W
            numeric_cfgs.append(cfg)
        return numeric_cfgs

    def test(self):
        cfg = self.get_batch_cfg(1)[0]
        subnet = self.cfg2subnet(cfg)
        client = LatencyMeasureClient(
                time_by_layer=True,
                remain_cache=True,
                )
        time, time_by_layer, res = client.model2latency(subnet, [3,224,224])
        print(time_by_layer)
        print(cfg)
        yaml.dump(res, open("save/test.yaml", "w"))


if __name__ == "__main__":
    fire.Fire(OFATask)
