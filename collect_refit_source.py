"""Fetch exactly the previously used public source; do not silently change mirrors."""
import json
from pathlib import Path
import requests
from src.refit_uncertainty import SOURCE, OUT, protocol, sha_bytes, load_source, write_json, require_registration

def main():
    spec, digest = protocol()
    require_registration()
    SOURCE.parent.mkdir(exist_ok=True)
    if SOURCE.exists():
        load_source()
        print('Reused exact pinned source')
        return
    r = requests.get(spec['source_url'], timeout=60)
    r.raise_for_status()
    if sha_bytes(r.content) != spec['source_payload_sha256']:
        raise RuntimeError('Pinned payload changed; stop, do not try another source')
    SOURCE.write_bytes(r.content)
    df = load_source()
    write_json(OUT/'source_receipt.json', {
        'source_url':spec['source_url'],'payload_sha256':sha_bytes(r.content),
        'payload_bytes':len(r.content),'rows':len(df),
        'parsed_rows_sha256':spec['source_parsed_rows_sha256'],
        'arm_counts':{str(k):int(v) for k,v in df.segment.value_counts().items()},
        'exact_duplicate_records_retained':int(df.drop(columns=['_row_id']).duplicated().sum()),
        'customer_identifier':'Unavailable; source-order key is not identity',
        'source_terms':'Public challenge data; raw dataset not redistributed in this release'})
    print('Pinned source verified:', len(df), 'customers')

if __name__ == '__main__': main()
