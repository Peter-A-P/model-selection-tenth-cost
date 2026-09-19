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

Azure Static Web Apps on the free tier, in the personal `PAP-POCs` subscription, resource group
`rg-portfolio`, East US 2: the same place `peterparker-ca`, `targeting-peterparker-ca` and
`capacity-peterparker-ca` already live. One Static Web App serves one set of files to every
hostname attached to it and does not route by host, so this page needs its own app rather than a
second hostname on the portfolio site's.

**The subscription is the step to get right.** The Azure CLI on this machine defaults to a
government work subscription, and nothing in a portfolio belongs there. Every command below
names the subscription, and the first one changes the default.

```powershell
az account set --subscription "PAP-POCs"
az account show --query name -o tsv          # must print PAP-POCs before anything is created

az staticwebapp create `
  --name adaptive-peterparker-ca `
  --resource-group rg-portfolio `
  --location "East US 2" `
  --sku Free `
  --subscription "PAP-POCs"
```

No repository is connected, so no workflow file and no Azure credential go into a repository
that is public. The deployment token is enough on its own to publish to that site: it is read
from the CLI straight into the environment of one shell, so it never appears on a command line
or in a shell history that is kept.

```powershell
$env:SWA_CLI_DEPLOYMENT_TOKEN = az staticwebapp secrets list `
  --name adaptive-peterparker-ca --resource-group rg-portfolio `
  --subscription "PAP-POCs" --query properties.apiKey -o tsv

npx --yes @azure/static-web-apps-cli deploy ./demo --env production
```

Run that from the repository root, after `uv run mselect demo build`, whenever `demo/data/` has
been rewritten. It needs Node and nothing else.

## The custom domain

A subdomain takes one CNAME. Only an apex needs the TXT validation, and this is a subdomain.

1. At Cloudflare, on the `peterparker.ca` zone, add:

   | Type | Name | Target | Proxy | TTL |
   |---|---|---|---|---|
   | CNAME | `adaptive` | the app's `<name>.azurestaticapps.net` hostname | **DNS only** | Auto |

   Proxied would block validation and the managed certificate. `targeting.peterparker.ca` is
   DNS only for that reason and this is the same.

2. Then, and not before, attach it. Azure validates by the CNAME alone, and issues and renews
   the certificate itself; there is nothing to buy or install.

   ```powershell
   az staticwebapp hostname set `
     --name adaptive-peterparker-ca --resource-group rg-portfolio `
     --hostname adaptive.peterparker.ca --subscription "PAP-POCs"
   ```

3. Check what the live host actually sends, which is the whole reason the policy is committed:

   ```powershell
   curl.exe -sI https://adaptive.peterparker.ca | Select-String "content-security-policy"
   curl.exe -s https://adaptive.peterparker.ca/data/index.json
   ```

   The first must print the policy from `staticwebapp.config.json`. The second must list six
   files. Then open the page and press **Run the test**: if the two charts stay empty, the data
   files did not deploy, and the page says so rather than showing an empty frame.

## The data files are stamped, and that is not decoration

`index.json` carries a hash of the bytes of every other payload, and the page reads that file
first and asks for the rest at `?v=<stamp>`. The hosting config tells the host not to cache
`index.json` and to cache everything else for an hour.

Without that pair, a deploy that changes the *shape* of a payload is served to browsers still
holding the previous one, and the page runs new code against an old file. That shipped on
2026-09-19: `panel.json` gained the fields naming each model, the page filtered on one of them,
both option groups came out empty, and every visitor from the previous hour got
"Cannot read properties of undefined" where the page should have been. A first load was fine,
which is why deploying and then checking the page did not catch it.

Two things to keep doing because of it. Rebuild the data whenever the page changes what it reads,
so the stamp moves with it. And when a deploy changes a payload's shape, check it in a browser
that already has the old one, not only in a fresh window.

## Order of operations

The README and the project's card on peterparker.ca both link to `adaptive.peterparker.ca`, so
the page goes up before those are pushed. A dead link in a public README is worse than a missing
one.

## If the free tier changes: GitHub Pages

Pages serves a subdirectory only from a branch root, so `demo/` has to become the root of a
published branch:

```bash
git subtree push --prefix demo origin gh-pages
```

Two things are worse that way: `staticwebapp.config.json` is ignored, so the content security
policy is lost, and the fallback rewrite goes with it.
