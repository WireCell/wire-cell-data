#!/usr/bin/env jsonnet

/**

This file yields a dictionary mapping a canonical detector name to its "detector
data object".  A detector data object provides data file names and high-level
information.

WCT configuration code is encouraged to include lines like:


  local detectors = import "detectors.jsonnet";
  local detdata = detectors.<name>;

Or, better, provide detector-independent configuration with a top-level argument
allowing the user to provide the canonical detector name.

  local detectors = import "detectors.jsonnet";
  function(detname)
      local det = detectors[detname];
      // ...

See wirecell.util.detectors for a Python interface and

  wcpy util detectors --help

For a CLI.

*/

// Explicitly define a length unit that is consistent with WCT's system of units
// in order not to import wirecell.jsonnet here.
local millimeter = 1.0;

// These are also supplied in the FR files and can be conveniently seen with:
//
// wcpy sigproc response-info <detname>
local hd_response_plane = 100.0*millimeter;
local vd_response_plane = 189.2*millimeter;

// A function to enforce the schema of each detector entry.  It transforms it's
// arguments into attributes of a dictionary keeping only non-null values.
local detector(detname,         // canonical detector name
               wires,           // wires file
               fields,          // field file(s), first is "nominal"
               noise=null,      // incoherent noise spectra
               wiregroups=null, // coherent groups of wires
               noisegroups=null, // coherent noise spectra
               chresp=null,      // per channel response
               qerr=null,        // charge error
               elresp=null,      // electronics response (if not analytical CE)
               // where the response plane is relative to the collection plane.
               response_plane=hd_response_plane) =
    std.prune({
        detname:detname,
        wires:wires,
        field: if std.type(fields) == "array" then fields[0] else fields,
        fields:if std.type(fields) == "array" then fields else [fields],
        noise:noise,
        chresp:chresp,
        qerr:qerr,
        elresp:elresp,
        response_plane:response_plane,
    });


// A temporary array - detname->object dict formed below.
local detectors = [
    // The "base" detector is idealized.  Someday replace these with idealized
    // equivalents but for now, copy from PDSP/uboone.
    detector("base",
             wires="protodune-wires-larsoft-v4.json.bz2",
             fields="dune-garfield-1d565.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
             qerr="microboone-charge-error.json.bz2", // reuse uboone
            ),
    detector("pdsp",
             wires="protodune-wires-larsoft-v4.json.bz2",
             fields="dune-garfield-1d565.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
             qerr="microboone-charge-error.json.bz2", // reuse uboone
            ),
    detector("uboone",
             wires="microboone-celltree-wires-v2.1.json.bz2",
             fields=["ub-10-half.json.bz2",
                     "ub-10-uv-ground-tuned-half.json.bz2",
                     "ub-10-vy-ground-tuned-half.json.bz2"], // array!
             noise="microboone-noise-spectra-v2.json.bz2",
             chresp="microboone-channel-responses-v1.json.bz2",
             qerr="microboone-charge-error.json.bz2", // reuse uboone
            ),
    detector("sbnd",
             wires="sbnd-wires-geometry-v0200.json.bz2",
             fields="garfield-sbnd-v1.json.bz2",
             noise="sbnd-noise-spectra-v1.json.bz2",
            ),
    detector("dune-vd",
             wires="dunevd10kt-1x6x6-3view30deg-wires-v1.json.bz2",
             fields="dunevd-resp-isoc3views-18d92.json.bz2",
             noise="dunevd10kt-1x6x6-3view30deg-noise-spectra-v1.json.bz2",
             qerr="microboone-charge-error.json.bz2", // reuse uboone
             response_plane=vd_response_plane,
            ),
    detector("dune-vd-coldbox",
             wires="dunevdcb1-3view-wires-v2-splitanode.json.bz2",
             fields="dunevd-resp-isoc3views-18d92.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2", // reuse pdsp
             elresp="dunevd-coldbox-elecresp-top-psnorm_400.json.bz2",
             response_plane=vd_response_plane,
            ),
    detector("dune10kt-1x2x6",
             wires="dune10kt-1x2x6-wires-larsoft-v1.json.bz2",
             fields="dune-garfield-1d60563.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
            ),
    detector("dunevd-crp2",
             wires="dunevdcrp2-wires-larsoft-v1.json.bz2",
             fields="dunevd-resp-isoc3views-18d92.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
             elresp="dunevd-coldbox-elecresp-top-psnorm_400.json.bz2",
             response_plane=vd_response_plane,
            ),
    detector("icarus",
             wires="icarus-wires-dualanode-v5.json.bz2",
             fields="garfield-icarus-fnal-rev1.json.bz2",
             chresp="icarus-channel-responses-v1.json.bz2",
             wiregroups="icarus_group_to_channel_map.json.bz2",
             noisegroups=[ "icarus_noise_model_int_by_board_TPCEE.json.bz2",
                           "icarus_noise_model_int_by_board_TPCEW.json.bz2",
                           "icarus_noise_model_int_by_board_TPCWE.json.bz2",
                           "icarus_noise_model_int_by_board_TPCWW.json.bz2",
                           "icarus_noise_model_coh_by_board_TPCEE.json.bz2",
                           "icarus_noise_model_coh_by_board_TPCEW.json.bz2",
                           "icarus_noise_model_coh_by_board_TPCWE.json.bz2",
                           "icarus_noise_model_coh_by_board_TPCWW.json.bz2" ],
            ),
    detector("iceberg",
             wires="iceberg-wires-larsoft-v2.json.bz2",
             fields="dune-garfield-1d565.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
            ),
    detector("pcbro",
             wires="pcbro-wires.json.bz2",
             fields="pcbro-response-avg.json.bz2",
             noise="protodune-noise-spectra-v1.json.bz2",
            ),
];

{[d.detname]:d for d in detectors}
