from allennlp.predictors.predictor import Predictor
import json
import multiprocessing as mp
import nli
import os


def _get_best_tags(srl):
    if len(srl['verbs']) == 0:
        # print(f"Sentence without tags: {' '.join(srl['words'])}")
        ret = ["O" for w in srl['words']]
        return ret
    elif len(srl['verbs']) == 1:
        return srl['verbs'][0]['tags']
    max_tags = 0
    final_tags = None
    for v in srl['verbs']:
        tag_count = 0
        for t in v['tags']:
            if t != 'O':
                tag_count += 1
        if tag_count > max_tags:
            final_tags = v['tags']
            max_tags = tag_count
    if final_tags is None:
        final_tags = srl['verbs'][0]['tags']

    return final_tags


def read_data(reader, predictor, tag_set):
    for ex in reader.read():
        label = ex.gold_label
        # srls = predictor.predict_batch_json([{"sentence": ex.sentence1}, {"sentence": ex.sentence2}])
        # ex.tags1 = _get_best_tags(srls[0])
        # ex.tags2 = _get_best_tags(srls[1])
        #
        # # Add all tags to the master list
        # tag_set.update(ex.tags1)
        # tag_set.update(ex.tags2)

        if (not reader.filter_unlabeled) or label != '-':
            yield ex
def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


def main():
    f_dir = "/Users/bszalapski/Documents/StanfordCourses/CS224U/cs224u_project/cs224uSNLI/data/nlidata"
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    batch = 200

    readers = [
        # nli.SNLITrainReader(os.path.join(f_dir, "snli_1.0"), samp_percentage=1, random_state=42),
        nli.SNLITestReader(os.path.join(f_dir, "snli_1.0"), samp_percentage=1, random_state=42),
        nli.AddamodTrainReader(f_dir, samp_percentage=1, random_state=42),
        # nli.AddamodDevReader(f_dir),
        nli.SubObjTrainReader(f_dir, samp_percentage=1, random_state=42),
        # nli.SubObjDevReader(f_dir),
        nli.BreakingSNLIReader(f_dir)
    ]
    for reader in readers:
        print("=" * 80)
        print(f"Processing reader {os.path.basename(reader.src_filename)}")

        # Initialize, from previous if available, the set of tags in the tag "vocabulary"
        tag_set = set()
        tag_file = "tags.txt"
        if os.path.isfile(tag_file):
            with open(tag_file, "r") as tp:
                tags = []
                for l in tp.readlines():
                    tags.append(l.strip())
                    tag_set = set(tags)

        o_file_name = f"preprocessed_{os.path.splitext(os.path.basename(reader.src_filename))[0]}.jsonl"
        o_path = os.path.join(os.path.dirname(reader.src_filename), o_file_name)
        total_ex = 0
        with open(o_path, "w") as fp:
            examples = []
            req_sents = []
            for ex in read_data(reader, predictor, tag_set):
                examples.append(ex)
                total_ex += 1
                # print(len(examples))
                if len(examples) % batch == 0:
                    for x in examples:
                        req_sents.append({"sentence": x.sentence1})
                        req_sents.append({"sentence": x.sentence2})
                    print(f"Processing a batch of {len(req_sents)}")
                    # print(req_sents)
                    srls = predictor.predict_batch_json(req_sents)
                    print("Extracting tags")
                    for (s1, s2), x in zip(grouped(srls, 2), examples):
                        # print(f"S1: {s1}, S2: {s2}")
                        x.tags1 = _get_best_tags(s1)
                        x.tags2 = _get_best_tags(s2)

                        # Add all tags to the master list
                        tag_set.update(x.tags1)
                        tag_set.update(x.tags2)
                        fp.write(
                            json.dumps({k: v for k, v in x.__dict__.items() if k in ["gold_label", "captionID",
                                                                                     "sentence1", "sentence2",
                                                                                     "tags1", "tags2"]}) + "\n")
                    print(f"Processed {total_ex} examples.")
                    examples = []
                    req_sents = []

        print(f"Preprocessed {total_ex} examples from {reader.src_filename}.\n\n")

        print(f"Number of unique tags found so far: {len(tag_set)}. Tags: {tag_set}")
        with open(tag_file, "w") as tp:
            for t in tag_set:
                tp.write(f"{t}\n")


if __name__ == "__main__":
    main()
    # predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    # snli_train = "/Users/bszalapski/Documents/StanfordCourses/CS224U/cs224u_project/cs224uSNLI/data/nlidata/snli_1.0/snli_1.0_train.jsonl"
    # pool_reader(snli_train, predictor)


# def chunkify(fname, size=1024*1024):
#     file_end = os.path.getsize(fname)
#     f = open(fname, 'r')
#     chunk_end = f.tell()
#     while True:
#         chunk_start = chunk_end
#         f.seek(f.tell() + size, os.SEEK_SET)
#         f.readline()
#         chunk_end = f.tell()
#         yield chunk_start, chunk_end - chunk_start
#         if chunk_start > file_end:
#             break
#     f.close()
#
#
# def read_lines(q, f_path, chunk_start, chunk_size, predictor):
#     print(f"In a worker with chunk_start{chunk_start}")
#     with open(f_path, "r") as f:
#         f.seek(chunk_start)
#         lines = f.read(chunk_size).splitlines()
#         sentences = []
#         for l in lines:
#             d = json.loads(l)
#             ex = nli.NLIExample(d)
#             sentences.append(f'{{"sentence": {ex.sentence1}}}, {{"sentence": {ex.sentence2}}}')
#
#         print(sentences)
#         srls = predictor.predict_batch_json([{"sentence": x.sentence1}, {"sentence": x.sentence2}] for x in sentences)
#         print(srls)
#         q.put(json.dumps({k: v for k, v in ex.__dict__.items() if k in ["gold_label", "captionID",
#                                                                                    "sentence1", "sentence2",
#                                                                                    "tags1", "tags2"]}))
#
#
# def listener(q, f_path):
#     out_file_dir = os.path.dirname(f_path)
#     out_file_end = os.path.basename(f_path)
#     out_file = os.path.join(out_file_dir, f"preprocessed_{out_file_end}")
#
#     fp = open(out_file, "w")
#     while 1:
#         m = q.get()
#         if m == 'kill':
#             break
#         fp.write(f"{m}\n")
#         fp.flush()
#     fp.close()
#
#
#
# def pool_reader(f_path, predictor):
#     manager = mp.Manager()
#     q = manager.Queue()
#     pool = mp.Pool(mp.cpu_count())
#
#     watcher = pool.apply_async(listener, (q, f_path))
#
#     jobs = []
#     for chunk_start, chunk_size in chunkify(f_path):
#         jobs.append(pool.apply_async(read_lines, (q, f_path, chunk_start, chunk_size, predictor)))
#         break
#
#     for job in jobs:
#         job.get()
#
#     q.put("kill")
#     pool.close()
