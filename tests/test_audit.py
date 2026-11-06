import unittest
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from lip_reading_claim_audit.phrases import LIPREADING, NOLIPREADING, GENERIC_NAMES
from lip_reading_claim_audit.audit import evidence, match_phrases, audit

EXPECTED_LIP = ('lip reading', 'lip-reading', 'lipreading', 'visual speech recognition', 'visual-speech recognition', 'reads lips', 'read lips', 'reading lips', 'reads lip movements', 'read lip movements', 'reading lip movements', 'lip reading support', 'supports lip reading', 'lip reading capability', 'lip reading feature', 'lip reading pipeline', 'lip reading system', 'lip reading tool', 'lip reading framework', 'lip reading library', 'lip reading api', 'lip reading model', 'lip reading models', 'lip reading engine', 'lip reading module', 'lip reading function', 'lip reading functionality', 'lip reading workflow', 'lip reading method', 'lip reading approach', 'lip reading service', 'lip reading results', 'lip reading performance', 'lip reading task', 'lip reading tasks', 'lip reading algorithm', 'lip reading algorithms', 'lip reading configuration', 'lip reading script', 'recognize speech from lips', 'recognizes speech from lips', 'recognizing speech from lips', 'speech recognition from lips', 'transcribe lip movements', 'transcribes lip movements', 'lip reading mode', 'lip movement recognition', 'lip-movement recognition', 'decode lip movements', 'convert lip movements to text')
EXPECTED_NO = ('does not read lips', 'do not read lips', "doesn't read lips", "don't read lips", 'no lip reading', 'without lip reading', 'never reads lips', 'does not support lip reading', 'lip reading not supported', 'lip reading unsupported', 'no lip reading support', 'lacks lip reading', 'missing lip reading', 'lip reading disabled', 'no lip reading model', 'without lip reading support', 'does not provide lip reading', 'no automatic lip reading', 'lip reading not available', 'no lip reading capability', 'does not recognize speech from lips', 'without a lip reading model', 'does not perform lip reading', 'never performs lip reading')
EXPECTED_GENERIC = ('read_lips', 'lip_read', 'lip_reading', 'recognize_lip_movements', 'decode_lip_movements', 'predict_lip_reading', 'transcribe_lips', 'lip_to_text', 'visual_speech_recognition', 'recognize_visual_speech', 'decode_visual_speech', 'predict_visual_speech', 'read_lip_movements', 'lip_movement_recognition', 'recognize_lip_reading', 'transcribe_lip_movements', 'decode_lips', 'predict_lips', 'classify_lip_movements', 'lip_speech_recognition', 'recognize_speech_from_lips', 'decode_speech_from_lips', 'transcribe_visual_speech', 'predict_lip_sequence')

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(LIPREADING, EXPECTED_LIP)
        self.assertEqual(NOLIPREADING, EXPECTED_NO)
        self.assertEqual((len(LIPREADING), len(NOLIPREADING)), (50, 24))

    def test_generic_oracle(self):
        self.assertEqual(GENERIC_NAMES, EXPECTED_GENERIC)
        self.assertEqual(len(GENERIC_NAMES), 24)

    def test_positive_routes(self):
        cases = [
            'import transformers as tf\ntf.pipeline("automatic-speech-recognition", model="AV-HUBERT/base")',
            'from transformers import pipeline as p\np("automatic-speech-recognition", model="av-hubert")',
            'import transformers\ntransformers.pipeline("automatic-speech-recognition", model="av-hubert")',
        ]
        for loader in ['import transformers as t\nm=t.AutoModelForCTC.from_pretrained("av-hubert")', 'from transformers import AutoModelForCTC as A\nm=A.from_pretrained("av-hubert")']:
            cases.extend([loader + '\nm.generate(x)', loader + '\nm(x)'])
        cases.extend([
            'from transformers import AutoModelForCTC as A\nA.from_pretrained("av-hubert").generate(x)',
            'from transformers import AutoModelForCTC as A\nA.from_pretrained("av-hubert")(x)',
        ])
        for method in ('predict', 'predict_classes'):
            cases.extend([
                'import lipnet as l\nm=l.LipNet()\nm.' + method + '(x)',
                'from lipnet.model import LipNet as L\nL().' + method + '(x)',
                'from lipnet import LipNet\nm=LipNet()\nm.' + method + '(x)',
            ])
        for name in EXPECTED_GENERIC:
            cases.extend(['object.' + name + '(x)', 'from package import ' + name + '\n' + name + '(x)', 'from package import ' + name + ' as f\nf(x)'])
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))

    def test_rejections(self):
        cases = [
            'import transformers, lipnet, cv2, mediapipe, speech_recognition',
            'import speech_recognition as sr\nr=sr.Recognizer()\nr.recognize_google(audio)',
            'import whisper\nm=whisper.load_model("base")\nm.transcribe(audio)',
            'from transformers import pipeline\npipeline("automatic-speech-recognition", model="openai/whisper")',
            'import mediapipe as mp\nm=mp.solutions.face_mesh.FaceMesh()\nm.process(image)',
            'import dlib\nd=dlib.get_frontal_face_detector()\nd(image)',
            'detector.detect_lip_landmarks(image)',
            'from wav2lip import Wav2Lip\nm=Wav2Lip()\nm.inference(audio, video)',
            'engine.say(text)\nengine.runAndWait()', 'vad.is_speech(audio)', 'spotter.detect_keyword(audio)',
            'from transformers import AutoModelForCTC\nm=AutoModelForCTC.from_pretrained("av-hubert")',
            'from transformers import AutoModelForCTC\nm=AutoModelForCTC.from_pretrained("whisper")\nm.generate(x)',
            'from lipnet import LipNet\nm=LipNet()',
            '"lip reading"\n# read lips\n"""reads lips"""',
            'from package import lip_read\nlip_read',
            'lip_read(x)',
            'from package import lip_read\ndef lip_read(x): pass\nlip_read(x)',
            'from lipnet import LipNet\nclass LipNet: pass\nm=LipNet()\nm.predict(x)',
            'from transformers import pipeline\ndef pipeline(*a, **k): pass\npipeline("automatic-speech-recognition", model="av-hubert")',
            'from transformers import AutoModelForCTC\nclass AutoModelForCTC: pass\nm=AutoModelForCTC.from_pretrained("av-hubert")\nm(x)',
            'import subprocess\nsubprocess.run(["lip_read", "video"])',
            'from lipnet import LipNet\nm=LipNet()', 'm.predict(x)',
            'from transformers import AutoModelForCTC\nm=AutoModelForCTC.from_pretrained("av-hubert")', 'm.generate(x)',
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def cli(self, root, *args):
        return subprocess.run([sys.executable, '-m', 'lip_reading_claim_audit', 'audit', str(root), *args], text=True, capture_output=True)

    def test_span_and_longest(self):
        negative, remaining = match_phrases('does not read lips', NOLIPREADING)
        self.assertEqual(negative, ['does not read lips'])
        self.assertEqual(match_phrases(remaining, LIPREADING)[0], [])
        for phrase in ('lip reading model', 'read lip movements'):
            self.assertEqual(match_phrases(phrase, LIPREADING)[0], [phrase])

    def test_case_and_human_collision(self):
        for text in ('LIP READING', 'she reads lips better than words'):
            root = self.fixture({'README.md': text})
            self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_fences_and_inline(self):
        root = self.fixture({'README.md': '```text\nlip reading\n```'})
        self.assertEqual(audit(root)[0], [])
        (root / 'README.md').write_text('`lip reading`')
        self.assertEqual(len(audit(root)[0]), 1)

    def test_cli_good(self):
        root = self.fixture({'README.md': 'lip reading', 'model.py': 'x.lip_read(video)'})
        result = self.cli(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_cli_bad_and_json(self):
        root = self.fixture({'README.md': 'lip reading'})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 2)
        finding = json.loads(result.stdout)[0]
        self.assertEqual(finding['id'], 'documented-lip-reading-claim-false')
        self.assertEqual(finding['severity'], 'HIGH')
        self.assertEqual(finding['verdict'], 'FAIL')
        self.assertTrue(finding['message'])

    def test_cli_test_only(self):
        root = self.fixture({'tests/fixture.md': 'lip reading'})
        self.assertEqual(self.cli(root).returncode, 0)
        result = self.cli(root, '--fail-on', 'medium', '--json')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)[0]['severity'], 'MEDIUM')

    def test_cli_denial(self):
        root = self.fixture({'README.md': 'does not read lips', 'a.py': 'x.lip_read(video)'})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 2)
        finding = json.loads(result.stdout)[0]
        self.assertEqual(finding['id'], 'documented-no-lip-reading-claim-false')
        self.assertEqual(finding['severity'], 'HIGH')

    def test_cli_no_claim_and_denial_without_code(self):
        for text in ('Neutral documentation', 'does not read lips'):
            root = self.fixture({'README.md': text})
            self.assertEqual(self.cli(root).stdout.strip(), 'PASS')

    def test_cli_version(self):
        result = subprocess.run([sys.executable, '-m', 'lip_reading_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')

    def test_cli_ignore(self):
        root = self.fixture({'docs/skip/claim.md': 'lip reading'})
        self.assertEqual(self.cli(root).returncode, 2)
        self.assertEqual(self.cli(root, '--ignore', 'docs/skip').returncode, 0)

    def test_split_modules_and_exclusions(self):
        root = self.fixture({'README.md': 'lip reading', 'a.py': 'from lipnet import LipNet\nm=LipNet()', 'b.py': 'm.predict(x)', '.venv/hidden.py': 'x.lip_read(v)'})
        self.assertEqual(self.cli(root).returncode, 2)

    def test_cli_errors_and_verbose(self):
        root = self.fixture({'README.md': 'Neutral'})
        self.assertIn('Scope:', self.cli(root, '--verbose').stderr)
        self.assertEqual(self.cli(root / 'absent').returncode, 1)
        self.assertEqual(self.cli(root, '--fail-on', 'wrong').returncode, 1)

    def test_readme_self_audit(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        root = self.fixture({'README.md': readme.read_text(encoding='utf-8')})
        self.assertEqual(audit(root)[0], [])
        self.assertEqual(self.cli(root).stdout.strip(), 'PASS')


if __name__ == '__main__':
    unittest.main()
