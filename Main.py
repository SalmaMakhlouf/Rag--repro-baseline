import subprocess, sys, argparse

def run(cmd):
    print('>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bm25_cfg', default='Configs/bm25.yaml')
    ap.add_argument('--hybrid_cfg', default='Configs/hybrid.yaml')
    args = ap.parse_args()

    run([sys.executable, 'src/index_bm25.py', '--config', args.bm25_cfg])
    run([sys.executable, 'src/dense_retriever.py', '--config', args.hybrid_cfg])
    run([sys.executable, 'src/rerank_ce.py', '--config', args.hybrid_cfg])
    run([sys.executable, 'src/eval_patk.py'])

if __name__ == '__main__':
    main()
