# Documentation claim auditor

An offline, deterministic, standard-library-only CLI that compares documentation
about mouth-movement decoding with static Python evidence. Requires Python 3.9
or newer. No network, telemetry, or file writes are performed by the auditor.

## Installation and usage

```sh
python3 -m pip install .
lip-reading-claim-audit audit .
python3 -m lip_reading_claim_audit audit . --json
lip-reading-claim-audit audit . --verbose --ignore docs/archive --ignore vendor --fail-on medium
lip-reading-claim-audit --version
python3 -m unittest discover
python3 -m build
```

The repository defaults to the current directory. Exit status 0 means no
findings meet the selected threshold; 2 means findings meet it; 1 means a usage
or IO error. The default threshold is high. JSON output is an array of findings,
each with verdict, message, identifier, severity, path, and matched phrases.
Verbose scope and per-file decisions go to standard error. Repeatable ignore
prefixes skip relative paths starting with the supplied text.

## Scope and rules

The auditor scans the root README and Markdown under docs. Triple-backtick
fenced blocks are excluded; inline backticks retain their content. Matching is
case-insensitive and uses exact substrings without stemming, fuzzy matching,
or hyphen normalization. Denial spans are removed before positive matching.
Longest-first matching counts each occurrence once under its longest phrase.
Test fixtures under tests receive medium severity only when positive claims
occur exclusively there. A denial without contradictory code passes.

Python sources are searched recursively except Git metadata, bytecode caches,
and virtual environments. Evidence must occur within one module. Imports and
aliases are resolved statically, and local functions or classes shadow imports.
The four routes are a configured inference pipeline, a loaded model followed
by inference, a constructed model followed by prediction, or one of the
24 recognized generic call names. Imports, comments, strings, uncalled names,
external subprocesses, and pairs split across modules do not suffice.

```text
documented-lip-reading-claim-false          HIGH (MEDIUM for test-only claims)
documented-no-lip-reading-claim-false       HIGH
```

## Differentiation

The neighboring audio tool checks turning audio into text. This auditor checks
mouth-movement decoding from video.

```text
speech-recognition-claim-audit
```

The neighboring face tool checks finding or identifying faces. This auditor
checks the checked capability rather than facial location or identity.

```text
face-claim-audit
```

The neighboring gesture tool checks classifying hand gestures. This auditor
checks speech decoding from mouth movements.

```text
gesture-recognition-claim-audit
```

## Limitations

Static patterns are not semantic judgment, and exact phrases do not capture
restatements. The following configured pipeline counts as evidence, although
its invocation alone does not prove the decoded text is correct.

```python
from transformers import pipeline
pipeline("automatic-speech-recognition", model="av-hubert/base")
```

The generic method below counts even on an unrelated object because the method
name is shared. The model construction and prediction pair below shows use of
the checked feature, without proving decoded text is correct.

```python
unrelated.lip_read(video)
from lipnet import LipNet
model = LipNet()
model.predict(video)
```

The generator named below deliberately does not count: it generates synced
video from audio and a face video instead of decoding speech from mouth
movements. Audio recognition, facial landmarks, synthesis, voice activity
and keyword detection also do not establish the checked capability.

```text
Wav2Lip
```

Exact substring matching cannot distinguish claims about software from human
skills. The expressions below can describe a person understanding someone
else's speech visually; they still match. This is an acknowledged ambiguity.

```text
read lips
reads lips
reading lips
lip reading
she reads lips better than words
```

Malformed Python and unreadable scoped files produce an error. The auditor
does not execute repository code, inspect model weights, or validate accuracy.

## License

MIT. See LICENSE.
