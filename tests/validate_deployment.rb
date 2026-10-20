# frozen_string_literal: true

require 'yaml'
require 'json'

root = File.expand_path('..', __dir__)
source = File.read(File.join(root, 'Sasha.Deployment.Protocol.vOmega.yaml'))
stream = Psych.parse_stream(source)
raise 'Expected one YAML document' unless stream.children.length == 1
visit = lambda do |node|
  if node.is_a?(Psych::Nodes::Mapping)
    keys = node.children.each_slice(2).map { |key, _value| key.value }
    raise 'Duplicate YAML key' unless keys.uniq.length == keys.length
  end
  Array(node.children).each { |child| visit.call(child) }
end
visit.call(stream)
yaml = YAML.safe_load(source)
json = JSON.parse(File.read(File.join(root, 'Sasha.Deployment.Protocol.vOmega.json')))
raise 'YAML and JSON differ' unless yaml == json
protocol = yaml.fetch('Sasha.Deployment.Protocol.vΩ')
status = protocol.fetch('document_status')
raise 'Unexpected runtime claim' unless status.fetch('runtime_implemented') == false && status.fetch('state_restored') == false
raise 'Missing unknown boundary' unless protocol.dig('interpretation_policy', 'unknown') == 'preserved'
%w[field_density over_interpretation_risk].each do |key|
  metric = protocol.fetch('self_check').fetch('on_boot').fetch(key)
  raise 'Unmeasured metric must remain null' unless metric.fetch('value').nil? && metric.fetch('measurement_method').nil?
end
protocol.fetch('modules_required').each do |entry|
  entry.each_value { |value| raise 'Missing implementation status' unless value.fetch('implementation_status').start_with?('not_implemented') }
end
raise 'Missing observable acceptance criteria' unless protocol.fetch('acceptance_criteria').all? { |item| item.key?('observable') && item.key?('failure') && item.fetch('status') == 'requires_consumer_test' }
puts '[PASS] deployment equivalence, unique keys, implementation boundaries and null metrics'
