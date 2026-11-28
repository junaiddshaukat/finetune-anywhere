# Community showcase

Models built by people learning and applying fine-tuning. Submit during an event or any time afterward. An entry is a builder's report, not an endorsement or independently verified benchmark.

## Submit your model

Allow about 15 minutes once your project, public demo and write-up are ready.

1. Fork this repository on GitHub.
2. Copy `showcase/_template/` to `showcase/community/your-slug/`, or use `showcase/qwen-local-hackathon-2026/your-slug/` for that event.
3. Complete `model.yaml` and `README.md` with actual results. The template is a blank submission form, not an example model.
4. Open a PR titled `Showcase: Your Project Name` and choose the showcase PR template.
5. Read the automated check and comment. Fix any errors on the same branch.
6. After merge, the tables update automatically.

A slug uses lowercase letters, numbers and hyphens. It must match `model.yaml`. The `event` field must equal its parent folder. If you fine-tuned, publish actual base-versus-tuned outputs from the same held-out prompt.

## Using git

These Bash commands ask for your values instead of including a fake team entry:

```bash
read -r -p 'Your GitHub username: ' GH_USER
read -r -p 'Your entry slug (lowercase, hyphens): ' SLUG
git clone "https://github.com/$GH_USER/finetune-anywhere.git"
cd finetune-anywhere
git switch -c "showcase/$SLUG"
cp -R showcase/_template "showcase/community/$SLUG"
```

Edit the two copied files in your text editor. For a hackathon entry, use the event folder instead of `community` in these commands and the YAML. Then:

```bash
git add "showcase/community/$SLUG"
git commit -m "Showcase: $SLUG"
git push -u origin "showcase/$SLUG"
```

Open the fork on GitHub, select Compare & pull request, target this repository's `main`, and choose the showcase template. You can append `?template=showcase.md` to the new-PR page URL. [GitHub's fork-to-PR guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) explains the screens.

## Without git

Fork on GitHub. In the fork, create a new branch. Open each of the two template files, copy its contents, then use Add file > Create new file with a path such as `showcase/community/your-chosen-slug/model.yaml`. Fill the form before saving. Repeat for README.md. GitHub creates folders from the typed path. Select Contribute > Open pull request and target this repository's main branch. No terminal is required.

## Rules

- One PR changes one entry folder only. Put general documentation changes in another PR.
- No model weights. If you fine-tuned, link to a public Hugging Face or ModelScope model. Files over 5 MB are rejected.
- Use HTTPS links. State hardware and runtime. Fine-tunes also state quantization and actual before/after outputs.
- Hackathon entries need a public demo video. Entries opened after 28 November 2026, 23:59 PKT (18:59 UTC), are labelled `late` and not judged. Moving them to `community/` is welcome.
- An automated URL warning is not a rejection. Some hosts block HEAD requests; check the link while signed out.

## Models

<!-- SHOWCASE:START -->

## community

No submissions yet. Be the first to share a model you trained and ran locally.

## qwen-local-hackathon-2026

No submissions yet. Be the first to share a model you trained and ran locally.

<!-- SHOWCASE:END -->
