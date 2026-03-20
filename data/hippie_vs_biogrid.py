from datasets import load_dataset


if __name__ == "__main__":
    bernett = load_dataset('Synthyra/bernett_gold_ppi')
    biogrid = load_dataset('Synthyra/BIOGRID', split='train')

    bernett['train'] = bernett['train'].map(lambda x: {'id': '_'.join(sorted([x['A'], x['B']]))}, num_proc=4)
    bernett['valid'] = bernett['valid'].map(lambda x: {'id': '_'.join(sorted([x['A'], x['B']]))}, num_proc=4)
    bernett['test'] = bernett['test'].map(lambda x: {'id': '_'.join(sorted([x['A'], x['B']]))}, num_proc=4)
    biogrid = biogrid.map(lambda x: {'id': '_'.join(sorted([x['A'], x['B']]))}, num_proc=4)

    biogrid_ids = set(list(biogrid['id']))
    bernett_ids = set(list(bernett['train']['id']) + list(bernett['valid']['id']) + list(bernett['test']['id']))

    intersections = biogrid_ids.intersection(bernett_ids)

    percent_overlap = len(intersections) / len(bernett_ids) * 100

    print(f'Number of interactions in BIOGRID: {len(biogrid_ids)}')
    print(f'Number of interactions in Bernett: {len(bernett_ids)}')
    print(f'Number of intersecting interactions: {len(intersections)}')
    print(f'Percent overlap: {percent_overlap:.2f}%')
