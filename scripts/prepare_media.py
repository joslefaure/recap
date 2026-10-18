"""Prepare the fixed CATER and AVE frame/audio inputs from released media."""
import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from scipy.io import wavfile
from recap.data import load_manifest


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=('cater','ave-train','ave-test'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--source',type=Path,required=True,help='CATER/AVE training video directory, or released AVE.tar for test')
    args=p.parse_args()
    manifest=load_manifest(args.manifest)
    if args.task=='ave-test':
        with tarfile.open(args.source) as archive:
            def read(name):
                handle=archive.extractfile(name)
                if handle is None:
                    raise FileNotFoundError(name)
                return handle.read()
            for row in manifest['items']:
                write_new(args.data_root/row['audio'],read(row['audio_member']))
                for number,index in zip(row['frame_numbers'],row['frame_indices']):
                    payload=read(row['frames_prefix']+'frame_{:04d}.jpg'.format(index+1))
                    write_new(args.data_root/row['frame_pattern'].format(index=number),payload)
        return
    for row in manifest['items']:
        with tempfile.TemporaryDirectory(prefix='recap_media_') as directory:
            temp=Path(directory)
            if args.task=='cater':
                source=args.source/(row['id']+'.avi')
                pattern=temp/'frame_%02d.jpg'
                expression='+'.join('eq(n\\,{})'.format(i) for i in row['frame_indices'])
                command=['ffmpeg','-v','error','-i',str(source),'-vf',"select='{}'".format(expression),
                         '-vsync','vfr','-frames:v','32','-q:v','3',str(pattern)]
                subprocess.run(command,check=True)
                generated=[temp/'frame_{:02d}.jpg'.format(i) for i in range(1,33)]
            else:
                source=args.source/(row['source_id']+'.mp4')
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(source),
                    '-map','0:v:0','-vf','fps=10.0','-vsync','0','-frames:v','100','-q:v','2',str(temp/'frame_%03d.jpg'),
                    '-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(temp/'audio.wav')],check=True)
                generated=[temp/'frame_{:03d}.jpg'.format(index+1) for index in row['frame_indices']]
                rate,audio=wavfile.read(temp/'audio.wav')
                start,end=round(row['event_start']*rate),round(row['event_end']*rate)
                crop=audio[start:min(end,len(audio))]
                if rate!=16000 or not len(crop):
                    raise ValueError('Invalid AVE event audio')
                if end>len(audio):
                    print('Audio ends before event boundary: {} ({} samples)'.format(row['id'],end-len(audio)))
                buffer=io.BytesIO();wavfile.write(buffer,rate,crop)
                write_new(args.data_root/row['audio'],buffer.getvalue())
            if len(generated)!=len(row['frame_numbers']):
                raise ValueError('Frame count differs from manifest')
            for number,source in zip(row['frame_numbers'],generated):
                write_new(args.data_root/row['frame_pattern'].format(index=number),source.read_bytes())
        print(row['id'],flush=True)


if __name__=='__main__':
    main()
