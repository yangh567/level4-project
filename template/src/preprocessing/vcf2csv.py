"""
    This is the place to handle the vcf file,at moment is useless(DEPRECATED)
"""

vcf_path = './vcf_data/CosmicNonCodingVariants.normal.vcf'

import allel

allel.vcf_to_csv(vcf_path, vcf_path.replace('.vcf', '.csv'))
# callset = allel.read_vcf(vcf_path)
# print(callset.keys())
# pass

