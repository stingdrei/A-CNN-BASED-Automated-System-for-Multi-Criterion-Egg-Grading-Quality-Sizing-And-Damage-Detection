import { useEffect, useState } from 'react';
import { Settings as SettingsIcon, Save, Ruler, Info, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Input } from '../components/common/Input';
import { apiClient } from '../api/client';
import { UserSettings } from '../types';

export function Settings() {
  const [settings, setSettings] = useState<UserSettings>({ mm_per_pixel: 0.5 });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await apiClient.getSettings();
        setSettings(data);
      } catch (error) {
        console.error('Failed to fetch settings:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await apiClient.updateSettings(settings);
      setMessage({ type: 'success', text: 'Settings saved successfully!' });
    } catch (error) {
      setMessage({ type: 'error', text: getErrorMessage(error) });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Ruler className="w-5 h-5" />
            Camera Calibration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Millimeters per Pixel (mm/px)
            </label>
            <Input
              type="number"
              step="0.01"
              min="0.1"
              max="5"
              value={settings.mm_per_pixel}
              onChange={(e) => setSettings({ mm_per_pixel: parseFloat(e.target.value) || 0.5 })}
              placeholder="0.5"
            />
            <p className="text-sm text-gray-500 mt-2">
              This value converts pixel measurements to millimeters. 
              Adjust based on your camera setup and distance from eggs.
            </p>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="flex items-start gap-2">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">How to calibrate:</p>
                <ol className="mt-2 list-decimal list-inside space-y-1">
                  <li>Place a reference object of known size (e.g., a ruler) in the camera view</li>
                  <li>Measure the object in pixels using your camera software</li>
                  <li>Calculate: mm_per_pixel = actual_size_mm / measured_pixels</li>
                  <li>Enter the result above and save</li>
                </ol>
              </div>
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <AlertCircle className="w-4 h-4" />
              <span>Default value: 0.5 mm/px (suitable for ~15cm camera distance)</span>
            </div>
          </div>

          {message && (
            <div className={`p-3 rounded-lg ${message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
              {message.text}
            </div>
          )}

          <Button onClick={handleSave} isLoading={saving} className="flex items-center gap-2">
            <Save className="w-4 h-4" />
            Save Settings
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <SettingsIcon className="w-5 h-5" />
            About Calibration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-gray-600 space-y-3">
            <p>
              <strong>Why is calibration important?</strong>
            </p>
            <p>
              Accurate calibration ensures:
            </p>
            <ul className="list-disc list-inside space-y-1">
              <li>Correct size categorization (Small/Medium/Large)</li>
              <li>Accurate weight estimation</li>
              <li>Proper grade assignment (AA/A/B)</li>
            </ul>
            <p className="mt-3">
              <strong>Formula used:</strong>
            </p>
            <ul className="list-disc list-inside space-y-1">
              <li>Diameter (mm) = Pixels × mm_per_pixel</li>
              <li>Weight (g) = 0.05 × Diameter³ (clamped to 30-80g)</li>
              <li>Size: Small (&lt;50mm), Medium (50-60mm), Large (&gt;60mm)</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}