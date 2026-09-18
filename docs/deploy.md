# Putting the page on adaptive.peterparker.ca

The page is `demo/`: four static files, a `fonts/` directory and a `data/` directory. There is
no build step for the page itself, no backend and no dependency to install, so hosting it is a
file copy and a DNS record.

## What is in it

| File | What it is |
|---|---|
| `index.html` | The page and all of its prose. |
| `style.css` | peterparker.ca's palette and type, no framework. |
| `irt.js` | The adaptive test: the response model, the running posterior, the selector and the stopping rule, ported from `mselect/irt/model.py`, `mselect/cat/estimate.py` and `mselect/cat/select.py`. |
| `charts.js` | Every chart, drawn by hand into SVG and one canvas. No chart library. |
| `app.js` | Loading the data, the five controls, and the copy that changes with them. |
| `data/*.json` | Written by `uv run mselect demo build`. Never hand-edited. |
| `fonts/` | Inter and Newsreader, copied from peterparker.ca rather than linked to it, under the SIL Open Font License. |

About 250 KB of JSON, 180 KB of fonts and 60 KB of HTML, CSS and JavaScript.

## Rebuilding the data

```bash
uv run mselect demo build      # writes demo/data/*.json and prints the size of each
uv run mselect demo serve      # http://127.0.0.1:8081, under the headers the live site sends
```

`demo build` reads the frozen bank, the two simulations, the own-run validation and the three
experiments, which is the same set of artefacts `mselect report` reads for the README's table.
If a number on the page disagrees with a number in the README, one of the two was not rebuilt,
and `tests/test_demo.py` fails rather than letting the page drift: it checks the panel against
the validation file it claims to come from, and the page's list of data files against the files
the builder writes.

**Check the page with `mselect demo serve`, not with `python -m http.server`.** The second
sends no headers, so it shows a page the content security policy would partly refuse. That is
not hypothetical: project 01's colour swatches were blank on the live site for two weeks and
correct in every local check, because a style attribute in markup is what `style-src 'self'`
blocks. `tests/test_demo.py` now also asserts that no file in `demo/` contains one.

The adaptive test runs in the browser, so the page needs to be served rather than opened from
disk: `fetch` and ES modules both refuse a `file://` origin. The page says so if it happens.

## Hosting

`demo/staticwebapp.config.json` is committed and sets a policy that allows the page's own
script, stylesheet, fonts and JSON and nothing else, so the deployment cannot quietly start
loading anything off-origin.

Azure Static Web Apps on the free tier, which is what `targeting.peterparker.ca` uses for
project 01, and the two routes are the same:

**From this machine, putting nothing in the repository.** Create the Static Web App with
deployment source **Other**, copy its deployment token from the Overview blade, and deploy with
`npx @azure/static-web-apps-cli deploy ./demo --deployment-token <token> --env production`. The
token is enough on its own to publish to that site: it does not go in this repository, in
`CLAUDE.local.md`, or in a shell history that is kept.

**From GitHub, redeploying on every push.** Connect the repository and the `main` branch, choose
the **Custom** build preset, set the app location to `demo`, leave the API and output locations
empty. The output location matters: the page is already built, and pointing the deployment at a
build directory that does not exist is the usual way this fails.

Then add `adaptive.peterparker.ca` as a custom domain on that Static Web App and point a
Cloudflare `CNAME` at the app's default hostname, DNS only rather than proxied. One Static Web
App serves one set of files to every hostname attached to it, so this subdomain needs its own
resource rather than sharing the portfolio site's.
