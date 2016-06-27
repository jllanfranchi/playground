#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from xml.etree import ElementTree


def fmt_list(values, fmt, cwidth=14):
    return " ".join([format(x, fmt).rjust(cwidth) for x in values])


def extract_info(xml_file):
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    device_util = []
    gpu_ns_per_photon = []
    tot_ns_per_photon = []
    tot_photons_at_doms = []
    context_switches = []
    for item in root.find("I3XMLSummaryService").find("map").findall("item"):
        txt = item.find("first").text
        if "I3CLSimModule_makeCLSimHits_makePhotons_clsim_AverageDeviceTimePerPhoton" in txt:
            gpu_ns_per_photon.append(float(item.find("second").text))
        if "I3CLSimModule_makeCLSimHits_makePhotons_clsim_AverageHostTimePerPhoton" in txt:
            tot_ns_per_photon.append(float(item.find("second").text))
        if "I3CLSimModule_makeCLSimHits_makePhotons_clsim_DeviceUtilization" in txt:
            device_util.append(100*float(item.find("second").text))
        if "I3CLSimModule_makeCLSimHits_makePhotons_clsim_TotalNumPhotonsAtDOMs" in txt:
            tot_photons_at_doms.append(int(float(item.find("second").text)))
        if "context switches" in txt:
            context_switches.append(int(float(item.find("second").text)))

    fmt_float = "10.2f"
    fmt_int = "1.3e"
    print("  GPU time per photon: %s ns"
          % fmt_list(gpu_ns_per_photon, fmt=fmt_float))
    print("Total time per photon: %s ns"
          % fmt_list(tot_ns_per_photon, fmt=fmt_float))
    print("   Device utilization: %s pct"
          % fmt_list(device_util, fmt=fmt_float))
    print("Total photons at DOMs: %s photons"
          % fmt_list(tot_photons_at_doms, fmt=fmt_int))
    print("     Context switches: %s"
          % fmt_list(context_switches, fmt=fmt_int))


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Extract timings and benchmark info from XML file produced by
        benchmark.py and bare_particle_sim.py scripts''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'XML_FILE',
        default='benchmark.xml',
        help='Name of the file from which to extract timings'
    )
    args = parser.parse_args()

    extract_info(args.XML_FILE)
